import os.path as osp
from pathlib import Path

if __name__ == '__main__':

    root_dir = Path('~/storage/DCLDE/').expanduser()
    data_dir1 = root_dir / 'train_puli'
    data_dir2 = root_dir / 'whale_whistle'

    bin_files = [osp.basename(p).split('.')[0] for p in data_dir1.glob('*.bin')]
    bin_files2 = [osp.basename(p).split('.')[0] for p in data_dir2.glob('*/*.bin')]   
    # print(data_dir2)
    # print(bin_files2)
    for d in bin_files:
        if d not in bin_files2:
            print(d)