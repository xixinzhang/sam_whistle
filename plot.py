from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch
from sam_whistle.config import DWConfig, SAMConfig
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import tyro
import random

import sam_whistle.utils as utils


def plot_patch_grid():
    w, h = 15, 4
    space_pix = 10
    nrows, ncols = 3, 10
    subplot_width, subplot_height = 50, 50
    dpi = 300
    w += (ncols - 1) * space_pix / 300
    h += (nrows - 1) * space_pix / 300
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=space_pix / subplot_width, hspace=space_pix / subplot_height, bottom=0.1, left= 0.04)

    cfg = tyro.cli(DWConfig)
    dataset = WhistlePatch(cfg, 'train')
    sampled_data = random.sample(list(dataset), 240)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = fig.add_subplot(gs[i, j])
            img = sampled_data[idx]['img'][0]
            # print(img.min(), img.max())
            ax.imshow(img, cmap='bone', aspect="auto")
            # ax.axis("off")
            height, width = img.shape
            if j  == 0:
                ax.set_yticks([0, height-1])
                ax.set_yticklabels([f'{height}', '0' ])
                ax.tick_params(axis='y', labelsize=16)
            else:
                ax.set_yticks([])

            if i == nrows-1:
                ax.set_xticks([0, width-1])
                ax.set_xticklabels(['0', f'{width}'])
                ax.tick_params(axis='x', labelsize=16)
            else:
                ax.set_xticks([])
            
    
    plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.02)
    plt.savefig("imgs/patch_grid.png", dpi=fig.dpi, pad_inches=0)
    plt.show()

def plot_seg_grid():
    # fig, ax = plt.subplots(1, 1, figsize=(15, 3.61), dpi=300)

    cfg = tyro.cli(SAMConfig)
    dataset = WhistleDataset(cfg, 'test', spect_nchan=1)
    sample_idx = random.randint(0, len(dataset))
    sample_idx = 62
    # print(sample_idx)
    # img = dataset[sample_idx]['img'][0]*300 - 200
    # ax.imshow(img, cmap='bone',aspect="auto")
    # ax.set_position([0, 0, 1, 1])
    # ax.axis("off")
    # plt.savefig("imgs/seg_grid.png", dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    # plt.show()

    filename = 'seg_grid'
    save_dir = 'imgs'
    cmap = 'bone'
    dpi = 100
    img = dataset[sample_idx]['img'][0]
    img = (img - img.min()) / (img.max() - img.min())
    y_ticks_num = 5
    y_ticks = np.linspace(0, img.shape[0], num=y_ticks_num, endpoint=True, dtype=int)
    y_labels = y_ticks[::-1]
    x_tick_num = 6  
    x_ticks = np.linspace(0, img.shape[1], num=x_tick_num, endpoint=True, dtype=int)
    x_labels = x_ticks
    utils.visualize_array(img, filename, save_dir, cmap= cmap, left_margin_px=60, right_margin_px= 40, top_margin_px=10, bottom_margin_px=40, y_ticks_lables= [y_ticks, y_labels], x_ticks_lables= [x_ticks, x_labels], tick_size=16, label_size=20,  dpi=dpi)



def plot_nn():
    cfg = tyro.cli(SAMConfig)
    dataset = WhistleDataset(cfg, 'test', spect_nchan=1)

if __name__ == '__main__':
    # plot_patch_grid()
    plot_seg_grid()