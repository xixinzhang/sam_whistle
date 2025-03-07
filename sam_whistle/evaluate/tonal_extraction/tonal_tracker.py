import os
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import tyro
from scipy.interpolate import interp1d

from sam_whistle import utils
from sam_whistle.config import *
from sam_whistle.evaluate.tonal_extraction import ActiveSet, tfTreeSet
from sam_whistle.model import *


def exact_div(x, y):
    assert x % y == 0
    return x // y

WINDOW_MS = 8
HOP_MS = 2
SAMPLE_RATE = 192_000
N_FFT = exact_div(SAMPLE_RATE * WINDOW_MS,  1000)  
HOP_LENGTH = exact_div(SAMPLE_RATE * HOP_MS, 1000)

CHUNK_LENGTH = 3 # second cover whistle length
N_SAMPLES = int(SAMPLE_RATE * CHUNK_LENGTH)
N_FRAMES = exact_div(N_SAMPLES,  HOP_LENGTH) # 1500

FRAME_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)
NUM_FREQ_BINS = exact_div(N_FFT, 2) + 1
FREQ_BIN_RESOLUTION = SAMPLE_RATE / N_FFT # 125 Hz

TOP_DB = 80
AMIN = 1e-10


def load_audio(file:Union[str, Path]) -> torch.Tensor:
    """Load audio wave file to tensor
    
    Args:
        file: audio file path
    Return:
        waveform: (1, L)
    """
    try:
        waveform, sample_rate = torchaudio.load(file)
    except RuntimeError as e:
        import librosa
        waveform, sample_rate = librosa.load(file, sr=None)
        waveform = torch.tensor(waveform).unsqueeze(0)
    waveform =waveform/ torch.max(torch.abs(waveform)) # normalize to [-1, 1]
    return waveform, sample_rate  # (1, L)


def spectrogram(waveform: torch.Tensor, device: Optional[Union[str, torch.device]] = None):
    """Compute spectrogram from waveform
    
    Args:
        waveform: (1, L)
        device: device to run the computation

    Return:
        spectrogram: (F, T) in range [-TOP_DB, 0]
    """

    window = torch.hann_window(N_FFT).to(device=device)
    spec = F.spectrogram(
        waveform,
        pad=0,
        window=window,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        power=2,
        normalized=False,
        center=True,
        pad_mode="reflect",
        onesided=True,
    )
    # spec = F.amplitude_to_DB(spec, multiplier=10, amin=AMIN, db_multiplier = 1, top_db = TOP_DB)
    # bound spect to [-TOP_DB, 0]
    spec = F.amplitude_to_DB(spec, multiplier=10, amin=AMIN, db_multiplier = torch.max(spec).log10(), top_db = TOP_DB)
    spec = spec[..., :-1].squeeze(0) # drop last frame
    spec = torch.flipud(spec) # flip frequency axis
    return spec # (F, T)  [-TOP_DB, 0]

def normalize_spec_img(spec: torch.Tensor) -> torch.Tensor:
    """Normalize spectrogram to [0, 1] to be compatible with image format"""
    spec = (spec + TOP_DB) / TOP_DB
    return spec


@dataclass
class SpectConfig:
    crop: bool = True
    min_freq: int = 5000
    max_freq: int = 50000
    hop_ms: int = HOP_MS
    n_fft: int = N_FFT
    frame_ms: int = WINDOW_MS
    freq_bin: int = FREQ_BIN_RESOLUTION
    top_db: int = TOP_DB
    amin: float = AMIN
    center: bool = True
    split_ms: int = 3000
    block_size: int = 1500
    crop_bottom: int = int(min_freq // freq_bin)
    crop_top: int = int(max_freq // freq_bin)

class Config:
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    spect_cfg: SpectConfig = SpectConfig()
    start_s: int = 0
    end_s: float = np.inf
    click_thr_db: float = 10
    thre_norm: float = 0.5
    use_conf: bool = True
    order: int = 1
    select_method:Literal["simple", "simpleN"] = 'simple'
    minlen_ms: int = 150 # Whistles whose duration is shorter than threshold will be discarded.
    maxgap_ms: int = 50 # Maximum gap in energy to bridge when looking for a tonal
    maxslope_Hz_per_ms: int = 1000 # Maximum difference in frequency to bridge when looking for a tonal
    activeset_s: float = 0.05 # peaks with earliest time > activeset_s will be part of active_set otherwise part of orphan set
    peak_dis_thr: int = 2
    disambiguate_s: float = 0.3
    broadband:float = 0.01
    log_dir: str = 'logs/03-06-2025_15-49-03-sam_coco'


@dataclass
class TonalResults:
    dt_false_pos_all: int = 0
    dt_true_pos_all: int = 0
    dt_true_pos_valid: int = 0
    # gt 
    gt_matched_all: int = 0
    gt_missed_all: int = 0
    gt_matched_valid: int = 0
    gt_missed_valid: int = 0
    #
    all_deviation: list = field(default_factory=list)
    all_covered_s:list = field(default_factory=list)
    all_excess_s:list = field(default_factory=list)
    all_dura: list = field(default_factory=list)

    def merge(self, other):
        self.dt_false_pos_all += other.dt_false_pos_all
        self.dt_true_pos_all += other.dt_true_pos_all
        self.dt_true_pos_valid += other.dt_true_pos_valid
        self.gt_matched_all += other.gt_matched_all
        self.gt_missed_all += other.gt_missed_all
        self.gt_matched_valid += other.gt_matched_valid
        self.gt_missed_valid += other.gt_missed_valid
        self.all_deviation += other.all_deviation
        self.all_covered_s += other.all_covered_s
        self.all_excess_s += other.all_excess_s
        self.all_dura += other.all_dura


class TonalTracker:
    """Extract tonal from one wav file by graph search algorithm."""
    def __init__(self, cfg: TonalConfig, stem:str):
        self.cfg = cfg
        self.stem = stem
        # spectrogram parameters
        self.spect_cfg = self.cfg.spect_cfg
        self.offset_Hz = int(self.spect_cfg.crop) * self.spect_cfg.min_freq
        self.freq_bin = self.spect_cfg.freq_bin
        # spectrogram and GT tonals
        self.audio_dir = Path(cfg.root_dir) / 'audio'
        self.anno_dir = Path(cfg.root_dir) / 'anno'
        self.hop_s = self.spect_cfg.hop_ms / 1000
        self.block_size = self.cfg.spect_cfg.block_size
        self.spect_map, self.spect_snr = self._load_spectrogram(stem)  # [H, W]
        self.gt_tonals = self._load_gt_tonals(stem)
        # search parameters
        self.pre_peak_num = 0.25 * self.H
        self.maxslope_Hz_per_ms = self.cfg.maxslope_Hz_per_ms
        self.activeset_s = self.cfg.activeset_s
        self.minlen_s = self.cfg.minlen_ms / 1000
        self.maxgap_s = self.cfg.maxgap_ms / 1000

        # initialize
        self.current_win_idx = 0
        self.start_s = self.cfg.start_s
        self.current_s = self.cfg.start_s
        self.current_peaks = []
        self.current_peaks_freq = []
        self.active_set = ActiveSet()
        self.active_set.setResolutionHz(self.freq_bin)
        self.detected_tonals = None

        if self.cfg.use_conf:
            self.thre = self.cfg.thre_norm
            self._select_peaks = partial(self._select_peaks, method = self.cfg.select_method)
        else:
            self.thre = self.cfg.thre
            self._select_peaks = partial(self._select_peaks, method = self.cfg.select_method)
        # compare tonals
        # self.search_row = self.cfg.peak_tolerance_Hz // self.freq_bin

        # model
        if self.cfg.use_conf:
            assert self.cfg.log_dir is not None, "No model to load"
            self.log_dir = Path(self.cfg.log_dir)
    
    def reset(self):
        self.current_win_idx = 0
        self.current_s = self.cfg.start_s
        self.current_peaks = []
        self.current_peaks_freq = []
        self.active_set = ActiveSet()
        self.active_set.setResolutionHz(self.freq_bin)
        self.detected_tonals = None
            
    def _load_spectrogram(self, stem: str):
        """Load original spectrogram from wave file, used for graph search baseline with"""
        
        # Modified to load spectrogram segments as images [0, 1]
        audio_file = self.audio_dir / f'{stem}.wav'
        waveform, sample_rate = load_audio(audio_file) # [1, L] 
        wave_len_s = waveform.shape[-1] / sample_rate

        start_s = max(0, self.cfg.start_s)
        end_s = min(self.cfg.end_s, wave_len_s)
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        waveform = waveform[..., start_idx:end_idx]
        
        # to match the marie's implementation, set the center to False
        spect_power_db= spectrogram(waveform)# (freq, time) [-TOP_DB, 0]
        spect_power_db = normalize_spec_img(spect_power_db) # [0, 1]
        print(spect_power_db.shape, spect_power_db.min(), spect_power_db.max())
        
        self.origin_shape = spect_power_db.shape
        if self.cfg.spect_cfg.crop:
            spect_power_db = spect_power_db[-self.cfg.spect_cfg.crop_top: -self.cfg.spect_cfg.crop_bottom+1]
        
        print(spect_power_db.shape, spect_power_db.min(), spect_power_db.max())
        # spect_raw and spect_snr are useless in this case
        # self.spect_raw = np.flip(utils.normalize_spect(spect_power_db, method=self.cfg.spect_cfg.normalize, min=self.cfg.spect_cfg.fix_min, max= self.cfg.spect_cfg.fix_max), axis=0)
        self.H, self.W = spect_power_db.shape[-2:]
        spect_snr = np.zeros_like(spect_power_db)
        # block_size = self.block_size
        # for i in range(0, self.W, block_size):
        #     spect_snr[:, i: i+block_size] = utils.snr_spect(spect_power_db[:, i:i+block_size], self.cfg.click_thr_db, self.H * self.cfg.broadband)
        print(f'Loaded spectrogram from {stem}: {spect_power_db.shape} shape: {spect_power_db.shape}, min: {spect_power_db.min():.2f}, max: {spect_power_db.max():2f}')
        
        self.start_cols = self._get_blocks()
       
        if self.cfg.use_conf:
            # spect_power_db = np.flip(spect_power_db, axis=0)
            # spect_power_db = utils.normalize_spect(spect_power_db, method= self.cfg.spect_cfg.normalize,  min=self.cfg.spect_cfg.fix_min, max= self.cfg.spect_cfg.fix_max)
            print(spect_power_db.shape, spect_power_db.min(), spect_power_db.max())
            return spect_power_db, spect_snr
        else:
            return spect_snr, spect_snr

    def _load_gt_tonals(self, stem: str):
        """Load ground truth tonals from annotation file."""
        bin_file = self.anno_dir/ f'{stem}.bin'
        gt_tonals = utils.load_annotation(bin_file)
        return gt_tonals
    
    def _get_blocks(self,):
        block_size = self.block_size
        start_cols = list(range(0, self.W - block_size + 1, block_size))
        if start_cols[-1] < self.W - block_size:
            start_cols.append(self.W - block_size)
        
        return start_cols
    
    @torch.no_grad()
    def sam_inference(self,):
        assert self.cfg.use_conf, "Not using confidence model"
        cfg = SAMConfig(SpectConfig)
        model = SAM_whistle(cfg)
        model.to(cfg.device)
        # Load model weights
        if not cfg.freeze_img_encoder:
            model.img_encoder.load_state_dict(torch.load(os.path.join(self.log_dir, 'img_encoder.pth'), weights_only=True))
        if not cfg.freeze_mask_decoder:
            model.decoder.load_state_dict(torch.load(os.path.join(self.log_dir, 'decoder.pth'), weights_only=True))
        if not cfg.freeze_prompt_encoder:
            model.sam_model.prompt_encoder.load_state_dict(torch.load(os.path.join(self.log_dir, 'prompt_encoder.pth'), weights_only=True))

        # Inference
        model.eval()
        spect_map =self.spect_map
        pred_mask = np.zeros_like(spect_map)
        weights = np.zeros_like(spect_map)

        block_size = cfg.spect_cfg.block_size
        start_cols = self.start_cols

        for start in start_cols:
            end = start + block_size
            block = spect_map[..., start:end]  # (H, W)
            block = torch.tensor(np.stack([block, block, block], axis=0)).to(cfg.device).unsqueeze(0) # (1, 3, H, W)
            pred = model(block).cpu().numpy().squeeze()
            pred_mask[::-1, start:end] += pred
            weights[:, start:end] += 1
        
        pred_mask /= weights
        self.spect_map = pred_mask
    
    @torch.no_grad()
    def dw_inference(self,):
        assert self.cfg.use_conf, "Not using confidence model"
        cfg = DWConfig(PatchConfig)
        model = Detection_ResNet_BN2(cfg.width)
        model.to(cfg.device)

        # Load model weights
        if os.path.exists(os.path.join(cfg.log_dir, 'model_more.pth')):
            model.load_state_dict(torch.load(os.path.join(self.log_dir, 'model_more.pth'), map_location=cfg.device, weights_only = True))
        else:
            model.load_state_dict(torch.load(os.path.join(self.log_dir, 'model.pth'), map_location=cfg.device, weights_only = True))
        model.eval()

        # Inferences
        spect_map =self.spect_map
        assert spect_map.ndim == 2, "Spectrogram should be 2D"
        pred_mask = np.zeros_like(spect_map)
        weights = np.zeros_like(spect_map)

        patch_size = cfg.spect_cfg.patch_size
        stride = patch_size
        
        i_starts = torch.arange(0, self.H - patch_size + 1, stride) 
        j_starts = torch.arange(0, self.W - patch_size + 1, stride)

        if (self.H - patch_size) % stride != 0:
            i_starts = torch.cat([i_starts, torch.tensor([self.H - patch_size])])

        if (self.W - patch_size) % stride != 0:
            j_starts = torch.cat([j_starts, torch.tensor([self.W - patch_size])])

        i_grid, j_grid = torch.meshgrid(i_starts, j_starts, indexing='ij')
        patch_starts = torch.stack([i_grid.flatten(), j_grid.flatten()], dim=-1)

        for i, j in patch_starts:
            patch = spect_map[..., i:i+patch_size, j:j+patch_size]
            patch = torch.tensor(patch).unsqueeze(0).to(cfg.device).unsqueeze(0) # (1, 1, H, W)
            pred = model(patch).cpu().numpy().squeeze()
            pred_mask[i:i+patch_size, j:j+patch_size] += pred
            weights[i:i+patch_size, j:j+patch_size] += 1
        
        pred_mask /= weights            
        self.spect_map = pred_mask[::-1, :]



    @torch.no_grad()
    def fcn_spect_inference(self,):
        assert self.cfg.use_conf, "Not using confidence model"
        cfg = FCNSpectConfig(PatchConfig)
        model = FCN_Spect(cfg)
        model.to(cfg.device)

        # Load model weights
        if os.path.exists(os.path.join(cfg.log_dir, 'model_more.pth')):
            model.load_state_dict(torch.load(os.path.join(self.log_dir, 'model_more.pth'), map_location=cfg.device, weights_only = True))
        else:
            model.load_state_dict(torch.load(os.path.join(self.log_dir, 'model.pth'), map_location=cfg.device, weights_only = True))
        
        # Inferences
        model.eval()
        spect_map =self.spect_map
        pred_mask = np.zeros_like(spect_map)
        weights = np.zeros_like(spect_map)

        block_size = cfg.spect_cfg.block_size
        start_cols = self.start_cols
        
        model.init_patch_ls((spect_map.shape[-2], block_size))

        for start in start_cols:
            end = start + block_size
            block = spect_map[..., start:end]  # (H, W)
            block = torch.tensor(block).unsqueeze(0).to(cfg.device).unsqueeze(0) # (1, 1, H, W)
            pred = model(block).cpu().numpy().squeeze()
            pred_mask[::-1, start:end] += pred
            weights[:, start:end] += 1

        pred_mask /= weights            
        self.spect_map = pred_mask


    @torch.no_grad()
    def fcn_encoder_inference(self,):
        assert self.cfg.use_conf, "Not using confidence model"
        cfg = FCNEncoderConfig(PatchConfig)
        model = FCN_encoder(cfg)
        model.to(cfg.device)

        # Load model weights
        model.load_state_dict(torch.load(os.path.join(self.log_dir, 'model.pth'),map_location=cfg.device, weights_only=True))

        # Inferences
        model.eval()
        spect_map =self.spect_map
        pred_mask = np.zeros_like(spect_map)
        weights = np.zeros_like(spect_map)

        block_size = cfg.spect_cfg.block_size
        start_cols = list(range(0, self.W - block_size + 1, block_size))
        if start_cols[-1] < self.W - block_size:
            start_cols.append(self.W - block_size)
        
        model.init_patch_ls()

        for start in start_cols:
            end = start + block_size
            block = spect_map[..., start:end]
            block = torch.tensor(block).unsqueeze(0).to(cfg.device).unsqueeze(0) # (1, 1, H, W)
            pred = model(block).cpu().numpy().squeeze()
            pred_mask[::-1, start:end] += pred
            weights[:, start:end] += 1
        
        pred_mask /= weights            
        self.spect_map = pred_mask

    def _select_peaks(self, spectrum, method='simple', thre = 0.5, order = 1):
        """Detect peaks based on SNR and other criteria, handling broadband signals separately.
        
        Args:
            spectrum: (H,) spectrum to analyze
                frequency of spectrum must increase from top to bottom
            method: 'simple' or 'simpleN' for peak detection
            thre: SNR threshold for peak detection.
                default 0.5 followed Pu Li if spectrum is normalized
                default 9.8 followed Marie Roch if spectrum is not normalized
        """

        if method == 'simple':
            peaks, _ = utils.find_peaks_simple(spectrum)
        elif method == 'simpleN':
            peaks, _ = utils.find_peaks_simpleN(spectrum, order)
        # Remove peaks that don't meet SNR criterion
        peaks = [p for p in peaks if spectrum[p] >= thre]
        peaks = utils.consolidate_peaks(peaks, spectrum, min_gap=self.cfg.peak_dis_thr)
        peak_num = len(peaks)
        
        if peak_num > 0:
            increase = (peak_num - self.pre_peak_num) / self.H
            if increase > self.cfg.broadband:
                pass
            else:
                self.current_peaks = peaks
                self.current_peaks_freq = peaks * self.freq_bin + self.offset_Hz
                self.pre_peak_num = peak_num
                return peaks.tolist()
        return []


    def _prune_and_extend(self):
        peaks = self.current_peaks
        assert len(peaks) > 0, "No peaks to prune and extend"

        self.active_set.prune(self.current_s, self.minlen_s, self.maxgap_s)
        times = np.ones_like(peaks) * self.current_s
        freqs = self.current_peaks_freq
        dbs = self.spect_map[:, self.current_win_idx]
        phases = np.zeros_like(peaks)
        ridges = np.zeros_like(peaks)
        peak_list = tfTreeSet(times, freqs, dbs, phases, ridges)
        self.active_set.extend(peak_list, self.maxslope_Hz_per_ms, self.activeset_s)
    

    def build_graph(self):
        all_peaks = []
        assert self.thre > 1 if not self.cfg.use_conf else True, "Threshold must be greater than 1"
        while self.current_win_idx < self.W:
            found_peaks = self._select_peaks(self.spect_map[:, self.current_win_idx], thre=self.thre, order=self.cfg.order)
            if found_peaks:
                self._prune_and_extend()
                all_peaks.extend([(self.H - p, self.current_win_idx, ) for p in found_peaks])
            self.current_win_idx += 1
            self.current_s += self.hop_s
        # final prune
        self.active_set.prune(self.current_s + 2*self.maxgap_s,self.minlen_s, self.maxgap_s)
        return all_peaks


    def get_tonals(self):
        """Process and filter tonal signals based on given parameters.
        """
        tonals = []
        self.discarded_count = 0

        def tone_py(tone):
            res = []
            for t in tone:
                res.append([t.time, t.freq])
            return np.array(res)
        
        def stat_avg_nth_wait_times(tonal, n: int):
            """
            Calculate the average wait time between samples separated by n positions.
            """
            # Get time samples
            times = np.array(tonal.get_time())
            samples = times / 0.002
            loop_end = len(samples) - n
            if loop_end <= 0:
                raise ValueError("n is too large for the number of samples")
            
            wait_times = samples[n:] - samples[:-n]
            avg_wait_time = np.mean(wait_times)
            return avg_wait_time

        for subgraph in self.active_set.getResultGraphs():
            # Disambiguate the subgraph
            graph = subgraph.disambiguate(
                self.cfg.disambiguate_s,
                self.freq_bin,
                False,
                False
            )
            edges = graph.topological_sort()
            for edge in edges:
                tone = edge.content
                if tone.get_duration() > self.minlen_s:
                    if stat_avg_nth_wait_times(tone, 3) < 18:
                        tonals.append(tone_py(tone))
                else:
                    self.discarded_count += 1

        self.detected_tonals = tonals
        print(f'Extract {len(tonals)} tonals discard {self.discarded_count} tonals from {self.stem}')

        return tonals

    def compare_tonals(self)-> TonalResults:
        """Compare extracted tonals with ground truth tonals."""
        assert self.detected_tonals is not None, "No detected tonals to compare."
        # GT tonals
        gt_tonals = self.gt_tonals
        gt_num = len(gt_tonals)
        gt_ranges = np.zeros((gt_num, 2))
        gt_durations = np.zeros(gt_num)
        # Detected tonals
        dt_tonals = self.detected_tonals
        dt_num = len(dt_tonals)
        dt_ranges = np.zeros((dt_num, 2))
        dt_durations = np.zeros(dt_num)

        spect_end_s = self.W * self.hop_s
        for gt_idx, gt_tonal in enumerate(gt_tonals):
            tonal_start_s = gt_tonal[:, 0].min()
            tonal_end_s =  min(gt_tonal[:, 0].max(), spect_end_s)
            gt_ranges[gt_idx] = (tonal_start_s, tonal_end_s)
            gt_durations[gt_idx] = tonal_end_s - tonal_start_s

        for dt_idx, dt_tonal in enumerate(dt_tonals):
            dt_start_s = dt_tonal[:, 0].min()
            dt_end_s = dt_tonal[:, 0].max()
            dt_ranges[dt_idx] = (dt_start_s, dt_end_s)
            dt_durations[dt_idx] = dt_end_s - dt_start_s


        dt_false_pos_all = list(range(dt_num))
        dt_true_pos_all = []
        dt_true_pos_valid = []
        gt_matched_all = []
        gt_matched_valid = []
        gt_missed_all = []
        gt_missed_valid = []
        all_deviation = []
        all_covered_s = []
        all_excess_s = []
        all_dura = []
        # go through each ground truth tonal
        for gt_idx in range(gt_num):
            gt_tonal = gt_tonals[gt_idx]
            gt_tonal = gt_tonal[np.argsort(gt_tonal[:, 0])]
            gt_dura = gt_durations[gt_idx]
            if gt_dura < self.hop_s:
                # suspiciously short tonal, less than one hop/frame
                continue
            # check validation of gt_tonal using snr_spect
            gt_start_s, gt_end_s = gt_ranges[gt_idx]   
            gt_block_start = np.ceil((gt_start_s - self.start_s) /self.hop_s).astype(int) # ceil
            gt_block_end = np.floor((gt_end_s - self.start_s) / self.hop_s).astype(int) + 1# floor
            gt_block = self.spect_snr[:, gt_block_start: gt_block_end]
            # serarch range
            gt_block_ts = self.start_s + np.arange(gt_block_start, gt_block_end) * self.hop_s
            gt_t, gt_f = gt_tonal[:, 0], gt_tonal[:, 1]
            gt_t_unique_idx = np.unique(gt_t, return_index=True)[1]
            gt_t = gt_t[gt_t_unique_idx]
            gt_f = gt_f[gt_t_unique_idx]
            gt_f_interp_fn = interp1d(gt_t, gt_f, fill_value="extrapolate")
            gt_f_interp = gt_f_interp_fn(gt_block_ts)
            # serach neighborhood
            gt_row = np.rint(gt_f_interp / self.freq_bin).astype(int) - self.spect_cfg.crop_bottom # + 1
            search_row_low = np.minimum(np.maximum(gt_row - self.search_row, 0), self.H)
            search_row_high = np.maximum(np.minimum(gt_row + self.search_row, self.H), 0)
            assert len(search_row_low) == len(search_row_high) == gt_block.shape[-1]
            spect_search = [np.max(gt_block[l:h, i]).item() for i, (l,h)in enumerate(zip(search_row_low, search_row_high))] 
            sorted_search_snr = np.sort(spect_search)
            # check validation
            bound_idx = max(0, round(len(sorted_search_snr) * (1- self.cfg.ratio_above_snr))-1)
            gt_snr = sorted_search_snr[bound_idx]
            valid = (gt_snr > self.cfg.snr_db) & (gt_dura >= self.minlen_s)
            # overlap with detected tonals
            overlapped_idx = utils.get_overlapped_tonals((gt_start_s, gt_end_s), dt_ranges)
            covered_s = 0
            excess_s = 0
            deviations = []
            matched = False
            for ov_idx in overlapped_idx:
                dt_tonal = dt_tonals[ov_idx]
                dt_t, dt_f = dt_tonal[:, 0], dt_tonal[:, 1]
                dt_ov_idx = np.nonzero((dt_t >= gt_start_s) & (dt_t <= gt_end_s))[0]
                dt_ov_t = dt_t[dt_ov_idx]
                dt_ov_f = dt_f[dt_ov_idx]
                gt_ov_f_interp = gt_f_interp_fn(dt_ov_t)
                deviation = np.abs(gt_ov_f_interp - dt_ov_f)
                if len(deviation)> 0 and np.mean(deviation) <= self.cfg.match_tolerance_Hz:
                    matched = True
                    if ov_idx in dt_false_pos_all:
                        dt_false_pos_all.remove(ov_idx)
                    dt_true_pos_all.append(ov_idx)
                    if valid:
                        dt_true_pos_valid.append(ov_idx)

                    deviations.extend(deviation)
                    covered_s += dt_ov_t[-1] - dt_ov_t[0]
                    excess_s += max(0, gt_t[0] - dt_t[0]) + max(0, dt_t[-1] - gt_t[-1])

            if matched:
                gt_matched_all.append(gt_idx)
                if valid:
                    gt_matched_valid.append(gt_idx)
            else:
                gt_missed_all.append(gt_idx)
                if valid:
                    gt_missed_valid.append(gt_idx)
            if matched:
                gt_deviation = np.mean(deviations) 
                all_deviation.append(gt_deviation) 
                all_covered_s.append(covered_s)
                all_dura.append(gt_dura)
                all_excess_s.append(excess_s)
                # print(f'gt_idx: {gt_idx}, covered_s: {covered_s}, excess_s: {excess_s}, deviations: {gt_deviation}')
        self.gt_tonals_valid = [gt_tonals[i] for i in gt_matched_valid+gt_missed_valid]
        self.gt_tonals_missed_valid = [gt_tonals[i] for i in gt_missed_valid]

        tonal_stats = TonalResults(
            dt_false_pos_all = len(dt_false_pos_all),
            dt_true_pos_all = len(dt_true_pos_all),
            dt_true_pos_valid = len(dt_true_pos_valid),
            gt_matched_all = len(gt_matched_all),
            gt_missed_all = len(gt_missed_all),
            gt_matched_valid = len(gt_matched_valid),
            gt_missed_valid = len(gt_missed_valid),
            all_deviation = all_deviation,
            all_covered_s = all_covered_s,
            all_excess_s = all_excess_s,
            all_dura = all_dura
        )
        return tonal_stats




if __name__ == "__main__":
    stem = 'Qx-Dd-SCI0608-N1-060814-150255' # 5 tonals
    # stem = 'Qx-Dd-SCI0608-N1-060816-142812' # 213 tonals
    # stem = 'Qx-Dc-SC03-TAT09-060516-173000'
    # stem = 'QX-Dc-FLIP0610-VLA-061015-165000'
    # stem = 'Qx-Dc-CC0411-TAT11-CH2-041114-154040-s'
    for stem in ['Qx-Dd-SCI0608-N1-060814-150255']:
        cfg = tyro.cli(TonalConfig)
        tracker = TonalTracker(cfg, stem)
        # tracker.sam_inference()
        for thre in [9.5, 10]:
            tracker.reset()
            tracker.thre = thre
            tracker.build_graph()
            tonals = tracker.get_tonals()
            # print(tonals[0])
            res = tracker.compare_tonals()
            for k, v in asdict(res).items():
                if isinstance(v, list):
                    print(f'{k}: {len(v)}')
                else:
                    print(f'{k}: {v}')

