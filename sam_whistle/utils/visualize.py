from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import torch
import cv2
from sam_whistle import utils
# plt.switch_backend('agg')

def show_spect(spect:np.array, fig, save:str = None):
    ax = fig.gca()
    ax.imshow(spect, cmap='viridis')
    ax.axis('on')
    y_ticks = np.linspace(0, spect.shape[0] - 1, num=10)
    y_labels = np.round(np.linspace(5, 50, num=10), 2)
    x_ticks = np.linspace(0, spect.shape[1] - 1, num=10)
    x_labels = np.round(np.linspace(0, 3, num=10), 2)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)')
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
    ax.imshow(mask_image,cmap='viridis')

def visualize(spect, masks,pred_mask, save_path=None, idx=None):
    spect = spect.squeeze().cpu().numpy()
    masks = masks.squeeze().cpu().numpy()
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
    # fig, axs = plt.subplots(figsize=(width/100, height/100))
    # show_spect(spect, fig)
    # show_points(points[0], points[1], axs, shape=(height, width))
    # fig.savefig(f'{save_path}/spect_{idx}_prompt.png')
    # plt.close()


def toggle_visual(image_sets):
    """
    Toggle and visualize different image sets, switch between images using keys 1, 2, 3, 4.
    
    Args:
        image_sets (list of dict): A list where each entry is a dictionary containing images.
                                   Keys are the types of images (e.g., 'Spectrogram', 'Mask', 'Prediction', 'Prompt').
    """
    num_sets = len(image_sets)
    idx = 0  # Index of current image set
    image_types = ['Spectrogram', 'Mask', 'Prediction', 'Prompt']  # Image types to toggle within one set
    current_image_type = image_types[0]  # Start by showing 'Spectrogram'
    
    # Get the image dimensions for calculating the figure size
    image_shape = list(image_sets[0].values())[0].shape  # Get the shape of the first image in the first set (H, W)
    height, width = image_shape[:2]

    # Set figure size to match image resolution (same size as image)
    dpi = 100  # Dots per inch, you can adjust this
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    
    # Function to update the displayed image
    def update_image(idx, image_type):
        ax.clear()
        if image_type in image_sets[idx]:
            ax.imshow(image_sets[idx][image_type], cmap='gray')
            ax.set_title(f'{image_type} - Set {idx + 1}')
        else:
            ax.text(0.5, 0.5, f'{image_type} not available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.draw()

    # Function to handle key press events
    def on_key(event):
        nonlocal idx, current_image_type
        if event.key == 'right':  # Move to the next image set
            idx = (idx + 1) % num_sets
        elif event.key == 'left':  # Move to the previous image set
            idx = (idx - 1) % num_sets
        elif event.key == '1':  # Toggle to 'Spectrogram'
            current_image_type = 'Spectrogram'
        elif event.key == '2':  # Toggle to 'Mask'
            current_image_type = 'Mask'
        elif event.key == '3':  # Toggle to 'Prediction'
            current_image_type = 'Prediction'
        elif event.key == '4':  # Toggle to 'Prompt'
            current_image_type = 'Prompt'
        update_image(idx, current_image_type)

    # Display the first image
    update_image(idx, current_image_type)

    # Connect the key press event to the figure
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show the plot
    plt.show()

def load_images_from_folder(folder_path, image_types, range:slice = None):
    """
    Load images from a folder. Images are grouped based on a naming convention.

    Args:
        folder_path (str): Path to the folder containing images.
        image_types (list): List of image types (e.g., ['Spectrogram', 'Mask', 'Prediction', 'Prompt']).
        
    Returns:
        list of dict: A list where each dictionary contains images for the specified types.
    """
    image_sets = []
    files = sorted(os.listdir(folder_path))  # Sort filenames to maintain order
    group_names = list(set([int(f.split('_')[0]) for f in files]))  # Group based on the prefix
    group_names = sorted(group_names)  # Sort group names
    group_names = group_names[range] if range else group_names

    for group in group_names:
        image_set = {}
        for img_type in image_types:
            image_path = os.path.join(folder_path, f"{group}_{img_type}.png")
            if os.path.exists(image_path):
                img = cv2.imread(image_path, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                image_set[img_type] = img
        if image_set:  # Only append non-empty sets
            image_sets.append(image_set)
    return image_sets

def toggle_visual2(folder1, folder2, range):
    """
    Toggle and visualize images from two folders, switch between folders and image types using keys.

    Args:
        folder1 (str): Path to the first folder.
        folder2 (str): Path to the second folder.
    """
    # Image types to toggle
    image_types = ['1.raw', '2.pred_tonal', '3.gt_tonal', '4.pred_conf']
    
    # Load images from both folders
    image_sets_folder1 = load_images_from_folder(folder1, image_types, range= range)
    image_sets_folder2 = load_images_from_folder(folder2, image_types, range=range)
    
    # Combine into a single list of sets
    combined_image_sets = [image_sets_folder1, image_sets_folder2]
    current_folder_idx = 0  # Current folder index (0 for folder1, 1 for folder2)
    idx = 0  # Index within the current folder
    current_image_type = image_types[0]  # Default image type to display

    # Get the image dimensions for figure size
    example_image = list(combined_image_sets[current_folder_idx][idx].values())[0]
    height, width = example_image.shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Function to update the displayed image
    def update_image():
        ax.clear()
        current_image_set = combined_image_sets[current_folder_idx][idx]
        if current_image_type in current_image_set:
            ax.imshow(current_image_set[current_image_type], cmap='bone')
            ax.set_title(f'{current_image_type} - Folder {current_folder_idx + 1} - Set {idx * 1500}')
        else:
            ax.text(0.5, 0.5, f'{current_image_type} not available', ha='center', va='center', fontsize=12)
        ax.axis('off')
        plt.draw()

    # Function to handle key press events
    def on_key(event):
        nonlocal current_folder_idx, idx, current_image_type
        if event.key == 'right':  # Next set
            idx = (idx + 1) % len(combined_image_sets[current_folder_idx])
        elif event.key == 'left':  # Previous set
            idx = (idx - 1) % len(combined_image_sets[current_folder_idx])
        elif event.key == 'up':  # Toggle to next folder
            current_folder_idx = (current_folder_idx + 1) % 2
            # idx = 0  # Reset index when switching folders
        elif event.key == 'down':  # Toggle back to previous folder
            current_folder_idx = (current_folder_idx - 1) % 2
            # idx = 0
        elif event.key == '1':  # Toggle to 'Spectrogram'
            current_image_type = '1.raw'
        elif event.key == '2':  # Toggle to 'Mask'
            current_image_type = '2.pred_tonal'
        elif event.key == '3':  # Toggle to 'Prediction'
            current_image_type = '3.gt_tonal'
        elif event.key == '4':  # Toggle to 'Prompt'
            current_image_type = '4.pred_conf'
        update_image()

    # Initial display
    update_image()

    # Connect the key press event to the figure
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Show the plot
    plt.show()



FLOAT_TYPE = [np.float32, np.float64, torch.float32, torch.float64]
INT_TYPE = [np.uint8, np.uint16, np.int32, np.int64, torch.int32, torch.int64]

def visualize_array(
    array,
    filename=None,
    save_dir="outputs",
    points=None,
    point_kwargs=[{
        "color": "green",
        "marker": "*",
        "markersize": 5,
    }],
    boxes=None,
    box_kwargs=[{
        "edgecolor": "red",
        "linewidth": 2,
        "facecolor":(0,0,0,0),
        "fill": False,
        "zorder": 10,
    }],
    mask=None,
    class_colors={},
    mask_color = [255, 0, 102],
    random_colors=True,
    mask_alpha=0.6,
    left_margin_px=0,
    bottom_margin_px=0,
    x_ticks_lables=None,
    y_ticks_lables=None,
    cmap="viridis",
):
    """Visualizes an array (PyTorch tensor, NumPy array) with optional overlays (points, lines, shapes).

    Args:
        array (torch.Tensor or np.ndarray): The image data as a NumPy array or a PyTorch tensor.
        filename (str): The filename to save the visualization.
        save_dir (str): The directory to save the visualization.
        axis_padding (float): The padding around the image.
        cmap (str): The colormap to use for grayscale images.
        
        The value range should be [0, 1]. Can be mask, edge map, grayscale, or RGB image.
        points: (list of list of tuples): List of point sets to overlay on the image, origin: top left (row, col)
        boxes: (list of list of tuples): List of boxes to overlay on the image.
        mask: (np.ndarray): Segmentation mask to overlay on the image.
    """
    assert isinstance(array, (torch.Tensor, np.ndarray)), "Input must be a PyTorch tensor or NumPy array"
    array = array.squeeze()
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()

    if array.ndim == 3 and array.shape[0] <= 4:
        array = np.moveaxis(array, 0, -1)  # Convert CHW -> HWC

    # Ensure values are in the correct range [0, 1] for float types
    if array.dtype in FLOAT_TYPE and np.max(array) > 1:
        raise ValueError("[Visualization] Array of float values should be normalized to range [0, 1]")

    # Validate the array shape
    if array.ndim not in [2, 3]:
        raise ValueError("Input array must have 2 (grayscale) or 3 (HWC) dimensions")

    dpi = 100
    height, width = array.shape[:2]
    fig, ax = plt.subplots(
        figsize=(
            (width+ 2 * left_margin_px) / dpi ,
            (height + 2 * bottom_margin_px) / dpi  ,
        ),
        dpi=dpi,
    )
    left_margin_fraction = left_margin_px / (width + 2 * left_margin_px)
    bottom_margin_fraction = bottom_margin_px / (height + 2 * bottom_margin_px)
    
    # Handle grayscale or mask visualization (HW)
    if array.ndim == 2:
        # grayscale image
        if array.dtype in FLOAT_TYPE:
            ax.imshow(array, cmap=cmap, vmin=0, vmax=1)
        # segmentation mask
        elif array.dtype in INT_TYPE:
            ax.imshow(array, cmap=cmap, vmin=0, vmax=array.max())
        else:
            raise ValueError("Array dtype not supported")
    # Handle RGB or single-channel visualization (HWC)
    elif array.ndim == 3:  # RGB or multi-channel
        ax.imshow(array)
    else:
        raise ValueError("Array shape not supported")
        
    # Overlay the mask (ignore zero values)
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.squeeze()
        assert mask.shape[:2] == array.shape[:2], "Mask shape must match image shape"
        colored_mask = np.zeros((height, width, 4), dtype=np.float32)
        assert mask.dtype in INT_TYPE, "Mask must be an integer array"
        unique_classes = np.unique(mask)
        for cls in unique_classes:
            if cls == 0:
                continue  # Skip zero values (background)
            if cls in class_colors:
                color = class_colors[cls]
            elif random_colors:
                # Assign random color for classes not specified
                random_color = np.random.rand(3,)  # Random RGB color
                color = np.concatenate([random_color, np.array([mask_alpha])], axis=0)
            else:
                color = np.array(mask_color) / 255  # Default color
                color = np.concatenate([color, np.array([mask_alpha])], axis=0)
            colored_mask[mask == cls] = color.reshape(1, 1, 4)
        ax.imshow(colored_mask, alpha=mask_alpha)

    # point format: [(x1, y1), (x2, y2), ...] top down
    if points is not None:
        if len(points) > len(point_kwargs):
            point_kwargs = point_kwargs * len(points)
        for i, (point_group, kwargs) in enumerate(zip(points, point_kwargs)):
            row, col = zip(*point_group)
            ax.scatter(col, row, **kwargs)

    # box format: [xmin, ymin, h, w]
    if boxes is not None:
        if len(boxes) > len(box_kwargs):
            box_kwargs = box_kwargs * len(boxes)
        for i, (box_group, kwargs) in enumerate(zip(boxes, box_kwargs)):
            for box in box_group:
                x0, y0, w, h = box
                rect = Rectangle((x0, y0), w, h, **kwargs)
                ax.add_patch(rect)

    if not left_margin_px and not bottom_margin_px:
        ax.axis("off")
    else:
        if x_ticks_lables is not None :
            if isinstance(x_ticks_lables, list):
                x_ticks, x_labels = x_ticks_lables
                ax.set_xticks(x_ticks, x_labels)
            elif isinstance(x_ticks_lables, int):
                x_ticks = np.linspace(0, width, x_ticks_lables, endpoint=True, dtype=int)
                ax.set_xticks(x_ticks)
        if y_ticks_lables is not None:
            if isinstance(y_ticks_lables, list):
                y_ticks, y_labels = y_ticks_lables
                ax.set_yticks(y_ticks, y_labels)
            elif isinstance(y_ticks_lables, int):
                y_ticks = np.linspace(0, height, y_ticks_lables, endpoint=True, dtype=int)
                ax.set_yticks(y_ticks)

    plt.subplots_adjust(
        left=left_margin_fraction, right=1 - left_margin_fraction, top=1 - bottom_margin_fraction, bottom=bottom_margin_fraction
    )

    if filename is not None and save_dir is not None:
        save_path = f"{save_dir}/{filename}.png"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig(save_path)
        plt.close()
    else:
        plt.show()




def plot_spect(spect, filename=None, save_dir="outputs", cmap='bone', pix_tick=False):
    spect = utils.normalize_spect(spect, method='minmax')
    if not pix_tick:
        y_ticks = np.linspace(0, spect.shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = np.linspace(5, 50, num=10, endpoint=True, dtype=int)[::-1]
        x_ticks = np.linspace(0, spect.shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = np.round(np.linspace(0, 3, num=20, endpoint=True), 2)
    else:
        y_ticks = np.linspace(0, spect.shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = y_ticks[::-1]
        x_ticks = np.linspace(0, spect.shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = x_ticks
    visualize_array(spect, filename, save_dir, cmap= cmap, left_margin_px=30, bottom_margin_px=30, y_ticks_lables= [y_ticks, y_labels], x_ticks_lables= [x_ticks, x_labels])


def plot_mask_over_spect(spect, mask, filename=None, save_dir="outputs", cmap='bone', random_colors=False, pix_tick=False):
    spect = utils.normalize_spect(spect, method='minmax')
    if not pix_tick:
        y_ticks = np.linspace(0, spect.shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = np.linspace(5, 50, num=10, endpoint=True, dtype=int)[::-1]
        x_ticks = np.linspace(0, spect.shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = np.round(np.linspace(0, 3, num=20, endpoint=True), 2)
    else:
        y_ticks = np.linspace(0, spect.shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = y_ticks[::-1]
        x_ticks = np.linspace(0, spect.shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = x_ticks
    visualize_array(spect, filename, save_dir, cmap=cmap, mask = mask, random_colors=random_colors, mask_alpha=1,  left_margin_px=30, bottom_margin_px=30, y_ticks_lables= [y_ticks, y_labels], x_ticks_lables= [x_ticks, x_labels])

def plot_points_over_spect(spect, points, filename=None, save_dir="outputs", cmap='bone', random_colors=False, pix_tick=False):
    spect = utils.normalize_spect(spect, method='minmax')
    if not pix_tick:
        y_ticks = np.linspace(0, spect.shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = np.linspace(5, 50, num=10, endpoint=True, dtype=int)[::-1]
        x_ticks = np.linspace(0, spect.shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = np.round(np.linspace(0, 3, num=20, endpoint=True), 2)
    else:
        y_ticks = np.linspace(0, spect.shape[0] - 1, num=10, endpoint=True, dtype=int)
        y_labels = y_ticks[::-1]
        x_ticks = np.linspace(0, spect.shape[1] - 1, num=20, endpoint=True, dtype=int)
        x_labels = x_ticks
    visualize_array(spect, filename, save_dir, cmap=cmap, points=points,  point_kwargs=[{
        "c": "green",
        # "marker": "*",
        "s": 1,
    }],left_margin_px=30, bottom_margin_px=30, y_ticks_lables= [y_ticks, y_labels], x_ticks_lables= [x_ticks, x_labels], )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir1', type=str, default='logs/08-25-2024_21-45-41/predictions')
    parser.add_argument('--dir2', type=str, default='logs/08-25-2024_21-45-41/predictions')
    # parser.add_argument('--num_sets', type=int, default=50)
    parser.add_argument('--range', type=int, nargs=2, default=[0, 50])
    args = parser.parse_args()
    # output_dir = Path(args.dir)
    # set_num = len(list(output_dir.glob('*_gt.png')))
    # # sets = np.random.choice(set_num, args.num_sets, replace=False)
    # sets = np.arange(args.num_sets)
    # image_sets = []
    # for idx in tqdm(sets):
    #     image_set = {}
    #     image_set['Spectrogram'] = plt.imread(output_dir / f"spect_{idx}_raw.png")
    #     image_set['Mask'] = plt.imread(output_dir / f"spect_{idx}_gt.png")
    #     image_set['Prediction'] = plt.imread(output_dir / f"spect_{idx}_pred.png")
    #     # image_set['Prompt'] = plt.imread(output_dir / f"spect_{idx}_prompt.png")
    #     image_sets.append(image_set)

    toggle_visual2(args.dir1, args.dir2,slice(*args.range))

