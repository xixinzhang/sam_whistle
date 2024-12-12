import struct 
import numpy as np
from scipy.interpolate import interp1d, splev, splrep, LSQUnivariateSpline
from numpy.polynomial import Polynomial
import cv2
from skimage.morphology import skeletonize
from pathlib import Path    

def load_annotation(bin_file:Path)-> list[np.ndarray]:
    """Read the bin file and obtain annotations of each contour"""
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



if __name__ == "__main__":
    sample_bin = '/home/asher/Desktop/projects/sam_whistle/data/dclde/annotation/palmyra092007FS192-070924-210000.bin'
    annos = load_annotation(sample_bin)
    print(annos[0].shape, len(annos), annos[0].dtype)
    print(np.max([np.max(anno) for anno in annos]))
    spec_anno = anno_to_spect_point(annos[0])
    print(spec_anno.shape, spec_anno[:, 1].max(), spec_anno[:, 1].min())