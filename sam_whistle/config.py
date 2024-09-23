from dataclasses import dataclass
from typing import Optional, Literal
import torch
import tyro

@dataclass
class Args:
    project: Optional[str] = None
    exp_name: Optional[str] = None

    # Data
    path: str = '/home/asher/Desktop/projects/sam_whistle/data/dclde'
    preprocess: bool = False
    # audio transfroms
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    frame_ms: int = 8
    hop_ms: int = 2
    split_ms: int = 3000

    num_pos_points: int=10
    num_neg_points: int=30
    box_pad: Optional[int] = 5  # 3
    thickness: Optional[int] = 3  # 3
    sample_points: Literal["random", "box"] = 'box'
    interpolation: Literal["linear", "polynomial", "spline"] = 'linear'

    # Model
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_type: str = 'vit_h'
    sam_ckpt_path: str = '/home/asher//Desktop/projects/sam_whistle/checkpoints'
    freeze_img_encoder:bool = True
    freeze_prompt_encoder:bool = True
    freeze_mask_decoder:bool = True
    ann_iters: int = 10
    fintune_decoder_type: Literal["sam", "pu"] = 'sam'
    sam_decoder: bool = True
    
    # Training
    loss_fn: Literal["mse", "dice", "bce_logits"] = 'dice'
    batch_size: int = 48
    epochs: int = 100
    decoder_lr: float = 2e-5
    encoder_lr: float = 1e-5
    save_path: str = '/home/asher/Desktop/projects/sam_whistle/logs'
    
    # Evaluation
    evaluate: bool = False
    visualize_eval: bool = False


if __name__ == '__main__':
    args = tyro.cli(Args)
    print(args)