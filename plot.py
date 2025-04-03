from sam_whistle.datasets.dataset import WhistleDataset, WhistlePatch
from sam_whistle.config import DWConfig, SAMConfig
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
import numpy as np
import tyro
import random

import sam_whistle.utils as utils


def plot_patch_grid():
    # w, h = 15, 4
    # space_pix = 10
    # nrows, ncols = 3, 10
    # subplot_width, subplot_height = 50, 50
    # dpi = 300
    # w += (ncols - 1) * space_pix / 300
    # h += (nrows - 1) * space_pix / 300
    # fig = plt.figure(figsize=(w, h), dpi=dpi)
    # gs = GridSpec(nrows, ncols, figure=fig, wspace=space_pix / subplot_width, hspace=space_pix / subplot_height, bottom=0.1, left= 0.04)

    # cfg = tyro.cli(DWConfig)
    # dataset = WhistlePatch(cfg, 'train')
    # sampled_data = random.sample(list(dataset), 240)

    # for i in range(nrows):
    #     for j in range(ncols):
    #         idx = i * ncols + j
    #         ax = fig.add_subplot(gs[i, j])
    #         img = sampled_data[idx]['img'][0]
    #         # print(img.min(), img.max())
    #         ax.imshow(img, cmap='bone', aspect="auto")
    #         # ax.axis("off")
    #         height, width = img.shape
    #         if j  == 0:
    #             ax.set_yticks([0, height-1])
    #             ax.set_yticklabels([f'{height}', '0' ])
    #             ax.tick_params(axis='y', labelsize=16)
    #         else:
    #             ax.set_yticks([])

    #         if i == nrows-1:
    #             ax.set_xticks([0, width-1])
    #             ax.set_xticklabels(['0', f'{width}'])
    #             ax.tick_params(axis='x', labelsize=16)
    #         else:
    #             ax.set_xticks([])
            

    
    # plt.subplots_adjust(left=0.02, right=0.99, top=0.98, bottom=0.02)
    # plt.savefig("imgs/patch_grid.png", dpi=fig.dpi, pad_inches=0)
    # plt.show()

    # Calculate dimensions to ensure square cells with equal spacing
    nrows, ncols = 3, 9
    cell_size = 0.5  # Size of each cell in inches (adjust as needed)
    h_space = v_space = 0.1  # Equal horizontal and vertical spacing in inches

    # Calculate the overall figure size needed for square cells
    w = ncols * cell_size + (ncols - 1) * h_space
    h = nrows * cell_size + (nrows - 1) * v_space

    dpi = 300
    w*= dpi/100
    h*= dpi/100
    fig = plt.figure(figsize=(w, h), dpi=dpi)

    # Create a gridspec with equal spacing in both directions
    main_gs = GridSpec(nrows, ncols, figure=fig,
                wspace=h_space/cell_size,  # Space relative to cell size
                hspace=v_space/cell_size,  # Same relative spacing
                left=0.03, right=0.98,      # Proportional margins
                bottom=0.1, top=0.98)      # Proportional margins

    cfg = tyro.cli(DWConfig)
    dataset = WhistlePatch(cfg, 'train')
    sampled_data = random.sample(list(dataset), 240)

    # Create all patch subplots
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            ax = fig.add_subplot(main_gs[i, j])
            axes.append(ax)
            
            img = sampled_data[idx]['img'][0]
            ax.imshow(img, cmap='bone', aspect="equal")  # Force equal aspect ratio
            ax.set_adjustable('box')  # Ensure the box aspect ratio is adjusted
            
            # Remove all axis spines, ticks, and labels
            ax.set_xticks([])
            ax.set_yticks([])

    # Now add arrows in separate axes outside the GridSpec
    # Create y-axis arrows (left side)
    for i in range(nrows):
        # Get position of the leftmost patch in this row
        patch_pos = axes[i * ncols].get_position()
        
        # Create a separate axis for the vertical arrow with consistent offset
        arrow_ax = fig.add_axes([patch_pos.x0 - 0.03, patch_pos.y0, 0.03, patch_pos.height])
        arrow_ax.set_xticks([])
        arrow_ax.set_yticks([])
        arrow_ax.axis('off')  # Turn off all axis elements
        
        # Create vertical arrow with consistent positioning
        vertical_arrow = FancyArrowPatch(
            (0.7, 0), (0.7, 1),
            arrowstyle='<->', linewidth=1, color='black',
            mutation_scale=8,
            transform=arrow_ax.transAxes
        )
        arrow_ax.add_patch(vertical_arrow)
        
        # Add centered label for height
        arrow_ax.text(0.35, 0.5, f'6.25 kHz', 
                verticalalignment='center', 
                fontsize=11, transform=arrow_ax.transAxes, rotation=90)

    # Create x-axis arrows (bottom)
    for j in range(ncols):
        # Get position of the bottom patch in this column
        patch_pos = axes[(nrows-1) * ncols + j].get_position()
        
        # Create a separate axis for the horizontal arrow with consistent offset
        arrow_ax = fig.add_axes([patch_pos.x0, patch_pos.y0 - 0.05, patch_pos.width, 0.05])
        arrow_ax.set_xticks([])
        arrow_ax.set_yticks([])
        arrow_ax.axis('off')  # Turn off all axis elements
        
        # Create horizontal arrow with consistent positioning
        horizontal_arrow = FancyArrowPatch(
            (0, 0.7), (1, 0.7),
            arrowstyle='<->', linewidth=1, color='black',
            mutation_scale=8,
            transform=arrow_ax.transAxes
        )
        arrow_ax.add_patch(horizontal_arrow)
        
        # Add centered label for width
        arrow_ax.text(0.5, -0.35, f'100 ms', 
                horizontalalignment='center', 
                fontsize=11, transform=arrow_ax.transAxes)

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
    # y_ticks_num = 5
    # y_ticks = np.linspace(0, img.shape[0], num=y_ticks_num, endpoint=True, dtype=int)
    # y_labels = y_ticks[::-1]
    # x_tick_num = 6  
    # x_ticks = np.linspace(0, img.shape[1], num=x_tick_num, endpoint=True, dtype=int)
    # x_labels = x_ticks
    y_ticks = np.linspace(0, img.shape[0], num=10, endpoint=True, dtype=int)
    y_labels = np.linspace(5, 50, num=10, endpoint=True, dtype=int)[::-1]
    x_ticks = np.linspace(0, img.shape[1], num=20, endpoint=True, dtype=int)
    x_labels = np.round(np.linspace(0, 3, num=20, endpoint=True), 2)
    utils.visualize_array(img, filename, save_dir, cmap= cmap, left_margin_px=60, bottom_margin_px=60, top_margin_px=10, right_margin_px=20 , y_ticks_lables= [y_ticks, y_labels], x_ticks_lables= [x_ticks, x_labels], dpi=dpi, x_label='Time (s)', y_label='Frequency (kHz)')
    # utils.visualize_array(img, filename, save_dir, cmap= cmap, left_margin_px=60, right_margin_px= 40, top_margin_px=10, bottom_margin_px=40, y_ticks_lables= [y_ticks, y_labels], x_ticks_lables= [x_ticks, x_labels], dpi=dpi, x_label='Time (s)', y_label='Frequency (kHz)', tick_size=16, label_size=20)



def plot_nn():
    cfg = tyro.cli(SAMConfig)
    dataset = WhistleDataset(cfg, 'test', spect_nchan=1)

if __name__ == '__main__':
    plot_patch_grid()
    # plot_seg_grid()