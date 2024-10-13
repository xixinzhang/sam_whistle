from dataclasses import dataclass
from typing import Optional, Literal
import torch
import tyro

@dataclass
class Args:
    project: Optional[str] = None
    exp_name: Optional[str] = None
    dev_mode: bool = False

    # Data
    path: str = '/home/asher/Desktop/projects/sam_whistle/data/dclde'
    preprocess: bool = False
    # audio transfroms
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    frame_ms: int = 8
    hop_ms: int = 2
    split_ms: int = 3000
    # crop
    crop: bool = True
    max_freq: int = 50000
    min_freq: int = 5000
    # cut to patches
    patch:bool = True
    patch_cached: bool = True
    balanced_cached: bool = True
    patch_size: int = 50
    patch_stride: int = 25
    slide_mean: bool = False

    # promt
    use_prompt:bool = False
    num_pos_points: int=10
    num_neg_points: int=30
    box_pad: Optional[int] = 5  # 3
    thickness: Optional[int] = 3  # 3
    sample_points: Literal["random", "box"] = 'box'
    interpolation: Literal["linear", "polynomial", "spline"] = 'linear'
    
    num_workers: int = 8
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    model: Literal["sam", "pu", "fcn_spect", "fcn_encoder"] = 'sam'
    model_type: str = 'vit_b'
    sam_ckpt_path: str = '/home/asher//Desktop/projects/sam_whistle/checkpoints'
    freeze_img_encoder:bool = False
    freeze_prompt_encoder:bool = True
    freeze_mask_decoder:bool = False
    ann_iters: int = 10
    sam_decoder: bool = False

    # SAM Training
    loss_fn: Literal["mse", "dice", "bce_logits"] = 'dice'
    spect_batch_size: int = 2
    epochs: int = 50
    decoder_lr: float = 1e-4
    prompt_lr: float = 1e-5
    encoder_lr: float = 5e-6
    save_path: str = '/home/asher/Desktop/projects/sam_whistle/logs'
    
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

    # FCN Encoder
    fcn_encoder_lr: float = 5e-5
    fcn_decoder_lr: float = 3e-3



    # Evaluation
    evaluate: bool = False
    visualize_eval: bool = False
    single_eval: bool = True


if __name__ == '__main__':
    args = tyro.cli(Args)
    print(args)