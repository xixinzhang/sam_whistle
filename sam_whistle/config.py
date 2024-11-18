from dataclasses import dataclass, asdict
from typing import Optional, Literal
import torch
import tyro
import numpy as np

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
    amin: float = 1e-15
    # block
    split_ms: int = 3000
    block_size:int = split_ms // hop_ms
    block_multi: int = 5
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

@dataclass
class PatchConfig(SpectConfig):
    patch_size: int = 50
    patch_stride: int = 25
    cached_patches: bool = False
    balance_patches: bool = True
    slide_mean: bool = False

@dataclass
class PuConfig:
    # data
    spect_cfg: PatchConfig
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    all_data: bool = False
    preprocess: bool = False

    debug: bool = False
    exp_name: Optional[str] = None
    log_dir: str = 'logs'

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class Args:
    project: Optional[str] = None

    # cut to patches
    patch:bool = True

    # promt
    num_pos_points: int=10
    num_neg_points: int=30
    box_pad: Optional[int] = 5  # 3
    thickness: Optional[int] = 3  # 3
    sample_points: Literal["random", "box"] = 'box'
    

    # Model
    ann_iters: int = 10

    # SAM Training

    
    # PU Model & Training
    pu_width: int = 32
    pu_lr: float = 1e-3
    pu_batch_size: int = 64
    pu_adam_decay: float = 0.00001
    pu_scheduler_gamma: float = 0.1
    pu_scheduler_stepsize: int = 250000
    pu_iters: int = 600000
    pu_epochs: int = 400
    pu_model_path: str = '/home/asher/Desktop/projects/sam_whistle/logs/10-06-2024_15-20-41_pu/model_pu.pth'

    # FCN spect
    fcn_spect_batch = 2048
    random_patch_order: bool = False
    fcn_spect_epochs: int = 100

    # FCN Encoder
    fcn_encoder_lr: float = 1e-4
    fcn_decoder_lr: float = 1e-3
    fcn_encoder_epochs: int = 100


    # Evaluation
    evaluate: bool = False
    visualize_eval: bool = False
    single_eval: bool = True

@dataclass
class SAMConfig:
    # data
    spect_cfg: SpectConfig
    root_dir: str = 'data/dclde'
    meta_file: str = 'meta.json'
    all_data: bool = False
    preprocess: bool = False

    debug: bool = False
    exp_name: Optional[str] = None
    log_dir: str = 'logs'

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type: str = 'vit_b'
    ckpt_dir: str = 'sam_checkpoints'
    sam_decoder: bool = False
    freeze_img_encoder:bool = False
    freeze_mask_decoder:bool = False
    use_prompt:bool = False
    freeze_prompt_encoder:bool = True

    num_workers: int = 8
    batch_size: int = 2
    epochs: int = 50
    encoder_lr: float = 5e-6
    decoder_lr: float = 1e-4
    prompt_lr: float = 1e-5
    loss_fn: Literal["mse", "dice", "bce_logits"] = 'dice'


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
    minlen_ms: int = 150
    maxgap_ms: int = 50
    maxslope_Hz_per_ms: int = 1000
    activeset_s: float = 0.05
    peak_dis: int = 2
    minlen_s: float = minlen_ms/1000
    maxgap_s: float = maxgap_ms/1000
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
    log_dir: Optional[str]= None

if __name__ == '__main__':
    args = tyro.cli(SAMConfig)
    print(args)
    print(asdict(args))