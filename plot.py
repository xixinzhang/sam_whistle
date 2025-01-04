from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch
from sam_whistle.config import DWConfig
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import tyro
import random

def plot_patch_grid():
    w, h = 15, 4
    space_pix = 1
    nrows, ncols = 8, 30
    subplot_width, subplot_height = 50, 50
    dpi = 300
    w += (ncols - 1) * space_pix / 300
    h += (nrows - 1) * space_pix / 300
    fig = plt.figure(figsize=(w, h), dpi=300)
    print(w, h)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=space_pix / subplot_width, hspace=space_pix / subplot_height)

    cfg = tyro.cli(DWConfig)
    dataset = WhistlePatch(cfg, 'test')
    sampled_data = random.sample(list(dataset), 240)

    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = fig.add_subplot(gs[i, j])
            img = sampled_data[idx]['img']
            ax.imshow(img, cmap='bone', aspect="auto") 
            ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("imgs/patch_grid.png", dpi=fig.dpi, pad_inches=0)
    plt.show()

def plot_seg_grid():
    fig, ax = plt.subplots(1, 1, figsize=(15, 3.61), dpi=300)

    cfg = tyro.cli(DWConfig)
    dataset = WhistleDataset(cfg, 'test', spect_nchan=1)
    # sample_idx = random.randint(0, len(dataset))
    sample_idx = 59
    img = dataset[sample_idx]['img']
    ax.imshow(img, cmap='bone',aspect="auto")

    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    plt.savefig("imgs/seg_grid.png", dpi=fig.dpi, bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == '__main__':
    plot_patch_grid()
    plot_seg_grid()