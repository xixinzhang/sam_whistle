from dataclasses import dataclass
from typing import Optional
import tyro

@dataclass
class Args:
    project: Optional[str] = None
    exp_name: Optional[str] = None

    # Data
    path: str = 'data/dclde'
    preprocess: bool = False
    # audio transfroms
    n_fft: Optional[int] = None
    hop_length: Optional[int] = None
    frame_ms: int = 8
    hop_ms: int = 2
    split_ms: int = 3000

    num_pos_points: int=10
    num_neg_points: int=10


if __name__ == '__main__':
    args = tyro.cli(Args)
    print(args)