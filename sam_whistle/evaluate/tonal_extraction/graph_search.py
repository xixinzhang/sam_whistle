from collections import deque
import numpy as np
from pathlib import Path
import tyro

from sam_whistle.evaluate.tonal_extraction import tfTreeSet, ActiveSet
from sam_whistle.config import TonalConfig
from sam_whistle import utils
from functools import partial

class TonalTracker:
    """Extract tonal from wave file"""
    def __init__(self, cfg: TonalConfig, stem: str):
        self.cfg = cfg
        self.spect_cfg = self.cfg.spect_config

        self.audio_dir = Path(self.cfg.root_dir) / 'audio'
        self.offset_Hz = int(self.spect_cfg.crop) * self.spect_cfg.min_freq
        self.freq_bin = self.spect_cfg.freq_bin

        self.broadband = self.spect_cfg.broadband
        self.click_thr_db = self.spect_cfg.click_thr_db

        self.spect = self._load_spectrogram(stem)
        self.pre_peak_num = 0.25 * self.H
        self.maxslope_Hz_per_ms = self.cfg.maxslope_Hz_per_ms
        self.activeset_s = self.cfg.activeset_s
        self.minlen_s = self.cfg.minlen_s
        self.maxgap_s = self.cfg.maxgap_s

        self.current_win_idx = 0
        self.current_s = self.start_s
        self.current_peaks = []
        self.current_peaks_freq = []
        if self.cfg.normalized:
            self.select_peaks = partial(self.select_peaks, method = self.cfg.select_method,thre = self.cfg.thre_norm)
        else:
            self.select_peaks = partial(self.select_peaks, method = self.cfg.select_method, thre = self.cfg.thre)

        self.active_set = ActiveSet()
        self.active_set.setResolutionHz(self.freq_bin)
    
    def _load_spectrogram(self, stem):
        """Load spectrogram from wave file, used for graph search."""
        audio_file = self.audio_dir / f'{stem}.wav'
        waveform, sample_rate = utils.load_wave_file(audio_file) # [C, L] Channel first
        wave_len_s = waveform.shape[-1] / sample_rate

        self.start_s = max(0, self.cfg.start_s)
        self.end_s = min(self.cfg.end_s, wave_len_s)
        start_idx = int(self.start_s * sample_rate)
        end_idx = int(self.end_s * sample_rate)
        waveform = waveform[:, start_idx:end_idx]
        spect_power_db= utils.wave_to_spect(waveform, sample_rate, **vars(self.spect_cfg))
        spect_power_db = spect_power_db[0].numpy() # (freq, time)
        if self.spect_cfg.crop:
            spect_power_db = spect_power_db[self.spect_cfg.crop_bottom: self.spect_cfg.crop_top+1]
        self.H, self.W = spect_power_db.shape[-2:]
        if self.spect_cfg.snr_spect:
            block_size = self.spect_cfg.block_size
            for i in range(0, self.W, block_size):
                spect_power_db[:, i: i+block_size] = utils.snr_spect(spect_power_db[:, i:i+block_size], self.click_thr_db, self.H * self.broadband)
        print(f'Loaded spectrogram from {stem}: {spect_power_db.shape}')
        return spect_power_db
    
    def _load_confidence_map(self, stem):
        pass

    def select_peaks(self, spectrum, method='simple', thre = 0.5):
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
            peaks, _ = utils.find_peaks_simpleN(spectrum)
        peaks = [p for p in peaks if spectrum[p] >= thre]
        peaks = utils.consolidate_peaks(peaks, spectrum, min_gap=self.cfg.peak_dis)
        peak_num = len(peaks)
        if peak_num > 0:
            increase = (peak_num - self.pre_peak_num) / self.H
            if increase > self.broadband:
                pass
            else:
                self.current_peaks = peaks
                self.current_peaks_freq = peaks * self.freq_bin + self.offset_Hz
                self.pre_peak_num = peak_num
                return True
        
        return False            


    def prune_and_extend(self):
        peaks = self.current_peaks
        assert len(peaks) > 0, "No peaks to prune and extend"

        self.active_set.prune(self.current_s, self.minlen_s, self.maxgap_s)
        times = np.ones_like(peaks) * self.current_s
        freqs = self.current_peaks_freq
        dbs = self.spect[:, self.current_win_idx]
        phases = np.zeros_like(peaks)
        ridges = np.zeros_like(peaks)
        peak_list = tfTreeSet(times, freqs, dbs, phases, ridges)
        self.active_set.extend(peak_list, self.maxslope_Hz_per_ms, self.activeset_s)
    
    def process_current_win(self):
        found_peaks = self.select_peaks(self.spect[:, self.current_win_idx])
        if found_peaks:
            self.prune_and_extend()
    
    def advance_win(self,):
        self.current_win_idx += 1
        self.current_s += self.spect_cfg.hop_ms / 1000

    def process(self):
        while self.current_win_idx < self.W:
            self.process_current_win()
            self.advance_win()
        self.active_set.prune(self.current_s + 2*self.maxgap_s,self.minlen_s, self.maxgap_s)

    def get_tonals(self):
        """Process and filter tonal signals based on given parameters.
        """
        tonals = []
        self.discarded_count = 0

        def tone_py(tone):
            res = []
            for t in tone:
                res.append((t.time, t.freq))
            return res
        
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
        return tonals

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

if __name__ == "__main__":
    stem = 'Qx-Dd-SCI0608-N1-060814-150255'
    stem = 'Qx-Dd-SCI0608-N1-060816-142812'
    cfg = tyro.cli(TonalConfig)
    tracker = TonalTracker(cfg, stem)
    print(tracker.spect.shape, tracker.freq_bin, tracker.offset_Hz)
    tracker.process()
    tonals = tracker.get_tonals()
    print(len(tonals), tracker.discarded_count)
    print(tonals[0])

