from dataclasses import asdict, dataclass
from typing import Literal, Optional

import numpy as np
import torch
import tyro


@dataclass
class SpectConfig:
    pad: int = 0
    # ms based fft params
    frame_ms: int = 8
    hop_ms: int = 2
    freq_bin: int = 1000 // frame_ms
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    top_db: Optional[None] = None
    center: bool = True
    amin: float = 1e-20
    normalize: Literal['minmax', 'zscore', 'fixed_minmax'] = 'fixed_minmax'
    mean: Optional[float] = 0.5
    std: Optional[float] = 0.5
    fix_min: float = -200
    fix_max: float = 100
    transform: bool = True
    # block
    split_ms: int = 3000
    block_size:int = split_ms // hop_ms
    # crop
    crop: bool = True
    max_freq: int = 50000
    min_freq: int = 5000
    crop_bottom: int = min_freq // freq_bin
    crop_top: int= max_freq // freq_bin
    # GT mask
    skeleton: bool = False
    interp: Literal["linear", "polynomial", "spline"] = 'linear'
    origin_annos: bool = False
    kernel_size:int = 2
    balance_blocks: bool = False

@dataclass
class PatchConfig(SpectConfig):
    patch_size: int = 50
    patch_stride: int = 25
    cached_patches: bool = False
    balance_patches: bool = True

@dataclass
class SAMConfig:
    # data
    spect_cfg: SpectConfig
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    all_data: bool = False
    save_pre: bool = False

    debug: bool = False
    exp_name: Optional[str] = "sam"
    log_dir: str = 'logs'

    device: str = 'cuda:0'
    model_type: str = 'vit_b'
    ckpt_dir: str = 'sam_checkpoints'
    sam_decoder: bool = False
    freeze_img_encoder:bool = False
    freeze_mask_decoder:bool = False
    use_prompt:bool = False
    freeze_prompt_encoder:bool = True

    num_workers: int = 8
    batch_size: int = 2
    epochs: int = 36
    encoder_lr: float = 5e-6
    decoder_lr: float = 1e-4
    prompt_lr: float = 1e-5
    loss_fn: Literal["mse", "dice", "bce_logits"] = 'dice'


@dataclass
class DWConfig:
    # data
    spect_cfg: PatchConfig
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    all_data: bool = False
    save_pre: bool = False

    debug: bool = False
    exp_name: Optional[str] = "deep_whistle"
    log_dir: str = 'logs'

    device: str = 'cuda:0'
    num_workers:int = 8
    width: int = 32
    lr: float = 1e-3
    batch_size: int = 64
    adam_decay: float = 0.00001
    scheduler_gamma: float = 0.1
    scheduler_stepsize:int = 250000
    iter_num: int = 600000
    iter_num_more: int = 1200000

@dataclass
class FCNSpectConfig:
    spect_cfg: PatchConfig
    # data
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    all_data: bool = False
    save_pre: bool = False

    debug: bool = False
    exp_name: Optional[str] = "fcn_spect"
    log_dir: str = 'logs'

    device: str = 'cuda:0'
    num_workers:int = 8
    width: int = 32
    lr: float = 1e-3
    dw_batch = 64
    batch_size: int = 2
    adam_decay: float = 0.00001
    scheduler_gamma: float = 0.1
    scheduler_stepsize:int = 250000
    iter_num: int = 600000
    iter_num_more: int = 1200000
    random_patch_order: bool = False

@dataclass
class FCNEncoderConfig:
    spect_cfg: PatchConfig
    # data
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    all_data: bool = False
    save_pre: bool = False

    debug: bool = False
    exp_name: Optional[str] = "fcn_encoder"
    log_dir: str = 'logs'

    device: str = 'cuda:0'
    num_workers:int = 8
    width: int = 32
    lr: float = 1e-3
    batch_size: int = 2

    encoder_lr: float = 5e-6
    decoder_lr: float = 1e-4
    epochs: int = 100
    freeze_img_encoder: bool = False
    freeze_mask_decoder: bool = False


@dataclass
class TonalConfig:
    spect_cfg: SpectConfig
    # range of signal to process
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    start_s: float = 0
    end_s: float = np.inf
    debug: bool = False

    use_conf: bool = False
    click_thr_db: float = 10

    thre_norm: float = 0.5
    thre: float = 9.8
    select_method:Literal["simple", "simpleN"] = 'simple'
    order: int = 1
    minlen_ms: int = 150 # Whistles whose duration is shorter than threshold will be discarded.
    maxgap_ms: int = 50 # Maximum gap in energy to bridge when looking for a tonal
    maxslope_Hz_per_ms: int = 1000 # Maximum difference in frequency to bridge when looking for a tonal
    activeset_s: float = 0.05 # peaks with earliest time > activeset_s will be part of active_set otherwise part of orphan set
    peak_dis_thr: int = 2
    disambiguate_s: float = 0.3
    broadband:float = 0.01

    # evaluation
    blocklen_s:int = 5
    block_pad_s: float = 1.5
    peak_tolerance_Hz:int = 500
    match_tolerance_Hz: int = 350
    snr_db: float = 10
    ratio_above_snr = 0.3

    # model
    log_dir: Optional[str]= 'logs'
    
if __name__ == '__main__':
    args = tyro.cli(SAMConfig)
    print(args)
    print(asdict(args))