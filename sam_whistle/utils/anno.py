from collections import defaultdict
import glob
import json
import os
import struct 
import numpy as np
from scipy.interpolate import interp1d, splev, splrep, LSQUnivariateSpline
from numpy.polynomial import Polynomial
import cv2
from skimage.morphology import skeletonize
from pathlib import Path
from rich import print as rprint

from sam_whistle import utils
from sam_whistle.evaluate.tonal_extraction.read_bin import tonalReader

def load_annotation(bin_file:Path)-> list[np.ndarray]:
    """Read the bin file and obtain annotations of each contour"""
    if isinstance(bin_file, str):
        bin_file = Path(bin_file)
    data_format = 'dd'  # 2 double-precision [time(s), frequency(Hz)]
    num_dim = 2
    with open(bin_file, 'rb') as f:
        bytes = f.read()
        total_bytes_num = len(bytes)
        if total_bytes_num==0:
            print(f'{bin_file}: is empty')
            return
        cur = 0
        annos = []
        while True:
            # get the data length
            num_point = struct.unpack('>i', bytes[cur: cur+4])[0]
            format_str = f'>{num_point * data_format}'
            point_bytes_num = struct.calcsize(format_str)
            cur += 4
            # read the contour data
            data = struct.unpack(f'{format_str}', bytes[cur:cur+point_bytes_num])
            data = np.array(data).reshape(-1, num_dim)
            annos.append(data)
            cur += point_bytes_num
            if cur >= total_bytes_num:
                break
        print(f'Loaded {len(annos)} annotated whistles from {bin_file.stem}.bin')
        return annos  #[(time(s), frequency(Hz)),...]

def load_tonal_reader(bin_file: Path)-> list[np.ndarray]:
    """Load tonal annotations from a .ann file using the tonalReader class.

    Args:
        bin_file: Path to the .ann file

    Returns:
        List of tonal annotations
    """
    reader = tonalReader(bin_file)
    contours = reader.getTimeFrequencyContours()
    annos = []
    num_dim=2
    for i, data in enumerate(contours):
        data = np.array(data).reshape(-1, num_dim)
        data = get_dense_annotation(data)  # make the contour continuous
        annos.append(data)
    print(f"Loaded {len(contours)} annotated whistles from {os.path.basename(bin_file)}")
    return annos  # [(time(s), frequency(Hz)),...]

def get_dense_annotation(traj: np.ndarray, dense_factor: int = 10):
    """Get dense annotation from the trajectory to make it continuous and fill the gaps.

    Args:
        traj: trajectory of the contour  [(time(s), frequency(Hz))]: (num_points, 2)
    """
    time = traj[:, 0]
    sorted_idx = np.argsort(time)
    time = time[sorted_idx]
    freq = traj[:, 1][sorted_idx]
    length = len(time)

    start, end = time[0], time[-1]
    new_time = np.linspace(start, end, length * dense_factor, endpoint=True)
    new_freq = np.interp(new_time, time, freq)
    return np.stack([new_time, new_freq], axis=-1)

def anno_to_spect_point(anno, height = 769, hop_ms = 2, freq_bin = 125):
    """Convert annotation to spectrogram point
    
    Args:
        anno: (n, 2) [time(s), frequency(Hz)]
        hop_ms: hop time in ms
    Returns:
        spec_points (n, 2) [col, row]
    """
    spec_anno = np.zeros_like(anno)
    sorted_indices = np.argsort(anno[:, 0])
    anno = anno[sorted_indices]
    spec_anno[:, 0] = anno[:, 0] / (hop_ms / 1000) # x
    spec_anno[:, 1] = anno[:, 1] / freq_bin # y
    # float location
    spec_anno[:, 1] = height - spec_anno[:, 1] # row
    return spec_anno

def interpolate_anno_points(new_x, x, y, interp="linear"):
    if interp == "linear":
        new_y = np.interp(new_x, x, y)
    elif interp == "polynomial":
        poly_interp = Polynomial.fit(x, y, 3)
        new_y = poly_interp(new_x)
    elif interp == "spline":
        n = 8
        knots = x[n:-n:n]
        spline = LSQUnivariateSpline(x, y, t=knots)
        new_y = spline(new_x)
    else:
        raise ValueError("Interpolation method not supported")
    return new_y

def get_dense_anno_points(contour, interp="linear", origin = False):
    x, y = contour[:, 0], contour[:, 1]
    x_min, x_max = x.min(), x.max()
    x_order = np.argsort(x)
    x = x[x_order]
    y = y[x_order]

    length = len(x)
    if origin:
        new_x = x
        new_y = y
    else:
        new_x = np.linspace(x_min, x_max, length*10, endpoint=True)
        new_y = interpolate_anno_points(new_x, x, y, interp)
    return new_x, new_y


def get_colored_tonal_map(shape, contours, interp='linear', origin=False):
    mask = np.zeros((*shape, 3))
    for i, contour in enumerate(contours):
        new_x, new_y = get_dense_anno_points(contour, origin= origin, interp=interp)
        new_x = np.maximum(0, np.minimum(new_x, shape[-1]-1)).astype(int)
        new_y = np.maximum(0, np.minimum(new_y, shape[-2]-1)).astype(int)
        c = np.random.rand(3)*0.6 + 0.4
        mask[new_y, new_x] = c
    return mask

def get_tonal_mask(shape, contours, interp='linear', origin=False):
    mask = np.zeros(shape)
    for i, contour in enumerate(contours):
        new_x, new_y = get_dense_anno_points(contour, origin= origin, interp=interp)
        new_x = np.maximum(0, np.minimum(new_x, shape[-1]-1)).astype(int)
        new_y = np.maximum(0, np.minimum(new_y, shape[-2]-1)).astype(int)
        mask[new_y, new_x] = i + 1
    return mask

def dilate_mask(mask, kernel_size=3):
    """dilate the mask"""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    return dilated


def skeletonize_mask(mask):
    """skeletonize the binary mask"""
    mask = mask.astype(np.uint8)
    skeleton = skeletonize(mask)
    return skeleton.astype(np.uint8)

def check_annotations():
    classes = ['bottlenose', 'common', 'melon-headed','spinner']
    meta = defaultdict(list)
    for s in classes[:2]:
        root_dir = os.path.join(os.path.expanduser("~"),f'storage/DCLDE/whale_whistle/{s}')
        bin_files = glob.glob('*.bin', root_dir=root_dir)
        stems = []
        for bin in bin_files:
            gts = utils.load_annotation(os.path.join(root_dir, bin))
            wavform, sr = utils.load_wave_file(os.path.join(root_dir, bin.replace('.bin', '.wav')))
            duration = len(wavform) / sr
            if gts:
                min_time = np.min([np.min(gt[:, 0]) for gt in gts])
                max_time = np.max([np.max(gt[:, 0]) for gt in gts])
                if duration > max_time + 30 or min_time > 30:
                    rprint(f'{bin}: {min_time}, {max_time}, audio duration: {duration}')
                stems.append(bin.replace('.bin', ''))
        meta['test'].extend(stems)
    # for s in classes[2:]:
    #     root_dir = os.path.join(os.path.expanduser("~"),f'storage/DCLDE/whale_whistle/{s}')
    #     bin_files = glob.glob('*.bin', root_dir=root_dir)
    #     stems = []
    #     for bin in bin_files:
    #         gts = utils.load_annotation(os.path.join(root_dir, bin))
    #         if gts:
    #             stems.append(bin.replace('.bin', ''))
    #     meta['train'].extend(stems)
    # print(f"train: {len(meta['train'])}, test: {len(meta['test'])}")

    # with open('data/cross_species/meta.json', 'w') as f:
    #     json.dump(meta, f, indent=4)


if __name__ == "__main__":
    sample_bin = '/home/asher/Desktop/projects/sam_whistle/data/dclde/annotation/palmyra092007FS192-070924-210000.bin'
    annos = load_annotation(sample_bin)
    print(annos[0].shape, len(annos), annos[0].dtype)
    print(np.max([np.max(anno) for anno in annos]))
    spec_anno = anno_to_spect_point(annos[0])
    print(spec_anno.shape, spec_anno[:, 1].max(), spec_anno[:, 1].min())