from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def show_spect(spect:np.array, fig, save:str = None):
    ax = fig.gca()
    ax.imshow(spect[::-1], origin='lower',cmap='viridis')
    ax.axis('on')
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight')


def show_points(coords, labels, ax, marker_size=50, shape =None):
    height, width = shape
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    for points in [pos_points, neg_points]:
        points[:, 0] = np.minimum(points[:, 0], width-1)
        points[:, 1] = np.minimum(points[:, 1], height-1)
    ax.scatter(pos_points[:, 0], height - pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], height - neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_mask(mask, ax, random_color=False):
    """Show mask on ax
    Args:
        mask: (h, w)
        ax: plt.axis
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 102/255, 1])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image[::-1], origin='lower',cmap='viridis')

def visualize(spect, masks, points, pred_mask, save_path=None, idx=None):
    spect = spect.squeeze().cpu().numpy()
    masks = masks.squeeze().cpu().numpy()
    points = [p.squeeze().cpu().numpy() for p in points]
    pred_mask = pred_mask.squeeze().cpu().numpy()
    width, height = spect.shape[:2][::-1]


    fig, axs = plt.subplots(figsize=(width/100, height/100))
    show_spect(spect, fig)
    fig.savefig(f'{save_path}/spect_{idx}_raw.png')
    plt.close()
    fig, axs = plt.subplots(figsize=(width/100, height/100))
    show_spect(spect, fig)
    show_mask(masks, axs)
    fig.savefig(f'{save_path}/spect_{idx}_gt.png')
    plt.close()
    fig, axs = plt.subplots(figsize=(width/100, height/100))
    show_spect(spect, fig)
    show_mask(pred_mask, axs)
    fig.savefig(f'{save_path}/spect_{idx}_pred.png')
    plt.close()
    fig, axs = plt.subplots(figsize=(width/100, height/100))
    show_spect(spect, fig)
    show_points(points[0], points[1], axs, shape=(height, width))
    fig.savefig(f'{save_path}/spect_{idx}_prompt.png')
    plt.close()

def visualize_array(array, save_path=None, idx=None, name=None):
    width, height = array.shape[-2:][::-1]
    fig, axs = plt.subplots(figsize=(width/100, height/100))
    axs.imshow(array[0, 0])
    fig.savefig(f'{save_path}/{name}_{idx}.png')
    plt.close()

def toggle_visual(image_sets):
    """
    Function to visualize multiple sets of images, masks, and depth images 
    with keyboard toggles.
    
    Parameters:
    image_sets (list of dicts): A list of dictionaries where each dictionary contains
                                'image', 'mask', and 'depth' keys corresponding to each set.
    """
    # Initialize the current index for the image set and the current display type
    current_set_idx = 0
    current_display_type = 'Prediction'

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)  # Adjust the plot to fit the text

    # Function to update the display based on the current index and display type
    def update_display():
        ax.clear()
        data = image_sets[current_set_idx][current_display_type]
        im_display = ax.imshow(data, cmap='gray' if data.ndim == 2 else None)
        ax.set_title(f'Set {current_set_idx + 1} - {current_display_type.capitalize()}')
        plt.axis('off')
        plt.figtext(0.5, 0.05, 
                    "Press 1: Prediction | Press 2: Mask | Press 3: Prompt | Press 4: Spectrogram\n Left/Right Arrows: Change Set", 
                    ha="center", fontsize=12)
        fig.canvas.draw_idle()

    # Handle key press events
    def toggle_display(event):
        nonlocal current_display_type, current_set_idx

        if event.key == '1':
            current_display_type = 'Prediction'
        elif event.key == '2':
            current_display_type = 'Mask'
        elif event.key == '3':
            current_display_type = 'Prompt'
        elif event.key == '4':
            current_display_type = 'Spectrogram'
        elif event.key == 'left':
            current_set_idx = (current_set_idx - 1) % len(image_sets)
        elif event.key == 'right':
            current_set_idx = (current_set_idx + 1) % len(image_sets)

        update_display()

    fig.canvas.mpl_connect('key_press_event', toggle_display)

    # Initial display
    update_display()

    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='logs/08-25-2024_21-45-41/predictions')
    parser.add_argument('--num_sets', type=int, default=50)
    args = parser.parse_args()
    output_dir = Path(args.dir)
    set_num = len(list(output_dir.glob('*_gt.png')))
    # sets = np.random.choice(set_num, args.num_sets, replace=False)
    sets = np.arange(args.num_sets)
    image_sets = []
    for idx in tqdm(sets):
        image_set = {}
        image_set['Spectrogram'] = plt.imread(output_dir / f"spect_{idx}_raw.png")
        image_set['Mask'] = plt.imread(output_dir / f"spect_{idx}_gt.png")
        image_set['Prediction'] = plt.imread(output_dir / f"spect_{idx}_pred.png")
        image_set['Prompt'] = plt.imread(output_dir / f"spect_{idx}_prompt.png")
        image_sets.append(image_set)

    toggle_visual(image_sets)

