import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def sample_points_box(masks, n_pos=5, n_neg= 10, box_pad = None, thickness = None):
    """sample point prompt from mask.
    Args:
        masks:
        data: [contours: (n, 2)] (x, y), [bbox: (4,)] [x, y]
    Returns:
        prompts: (n, (points, labels))
    """
    assert n_pos or n_neg, 'n_pos or n_neg must be provided'
    # spect, bboxes, contours,_, _ = data
    height, width = masks[0].shape
    if n_neg is None:
        n_neg = n_pos
    
    combined_mask = combine_masks(masks)
    all_pos_ids = np.argwhere(combined_mask == 1) # (y, x)
    all_mask_points = []
    all_mask_labels = []
    # for contour, bbox in zip(contours, bboxes):
    for mask in masks:
        assert mask.ndim == 2, 'mask must be 2D'
        pos_indices = np.argwhere(mask == 1) # (y, x)

        contour = np.flip(pos_indices, axis=1) # (x, y)
        # sample positive points
        n_contour = len(contour)
        if n_contour > n_pos:
            stratum_size = n_contour//n_pos
            pos_pts = []
            for i in range(n_pos):
                start_idx = i * stratum_size
                if i == n_pos - 1:
                    end_idx = n_contour
                else:
                    end_idx = (i + 1) * stratum_size
                
                # print(stratum_size, start_idx, end_idx)
                # Sample one element from the stratum
                stratum = contour[start_idx:end_idx]
                sampled_point = stratum[np.random.randint(0, len(stratum))]
                pos_pts.append(sampled_point)
        else:
            pos_pts = contour

        # sample negative points
        # expanded_bbox
        if box_pad:
            x_min, y_min, x_max, y_max = contour[:, 0].min(), contour[:, 1].min(), contour[:, 0].max(), contour[:, 1].max()
            x_min = x_min - box_pad if x_min - box_pad >0  else 0
            y_min = y_min - box_pad if y_min - box_pad >0  else 0
            x_max = x_max + box_pad if x_max + box_pad < width else width
            y_max = y_max + box_pad if y_max + box_pad < height else height
            box_h, box_w = y_max - y_min, x_max - x_min

            sides = []
            perimeter = 0
            # Determine which sides to include based on image constraints
            if x_min > 0:
                sides.append(('left', box_h))
                perimeter += box_h
            if y_min > 0:
                sides.append(('top', box_w))
                perimeter += box_w
            if x_max < width:
                sides.append(('right', box_h))
                perimeter += box_h
            if y_max < height:
                sides.append(('bottom',box_w))
                perimeter += box_w

            if perimeter == 0:
                raise ValueError("No sides available for sampling.")
            
            points_per_side = {side: int(np.round(n_neg * length / perimeter)) for side, length in sides}
            remaining_points = n_neg - sum(points_per_side.values())
            # print(points_per_side)
            # Distribute remaining points evenly
            for i in range(remaining_points):
                points_per_side[sides[i % len(sides)][0]] += 1
            
            # Generate points for each side
            if 'left' in points_per_side:
                pts = np.linspace(y_max, y_min, points_per_side['left'] + 1)
                left_side = [(x_min, y) for y in pts[:-1]]
            else:
                left_side = []
            if 'top' in points_per_side:
                pts = np.linspace(x_min, x_max, points_per_side['top'] + 1)
                top_side = [(x, y_min) for x in pts[:-1]]
            else:
                top_side = []
            if 'right' in points_per_side:
                pts = np.linspace(y_min, y_max, points_per_side['right'] + 1)
                right_side = [(x_max, y) for y in pts[:-1]]
            else:
                right_side = []
            if 'bottom' in points_per_side:
                pts = np.linspace(x_max, x_min, points_per_side['bottom'] + 1)
                bottom_side = [(x, y_max) for x in pts[:-1]]
            else:
                bottom_side = []
            neg_pts = left_side + bottom_side + right_side + top_side

        elif thickness:
            neg_pts = []
            for p in pos_pts:
                x, y  = p
                y1 = y - thickness if 0 < y - thickness< height  else 0 
                y2 = y + thickness if 0 < y + thickness< height  else height
                neg_pts.append((x, y1))
                neg_pts.append((x, y2))

        neg_pts = np.array(neg_pts)
        for p in neg_pts:
            if p in np.flip(all_pos_ids, axis=1):
                neg_pts = np.delete(neg_pts, np.argwhere(np.all(neg_pts == p, axis=1)), axis=0)

        points = np.concatenate((pos_pts, neg_pts), axis=0)
        labels = np.concatenate((np.ones(len(pos_pts)), np.zeros(len(neg_pts))), axis=0)
        all_mask_points.append(points)
        all_mask_labels.append(labels)
    
    all_mask_points = np.concatenate(all_mask_points, axis=0)
    all_mask_labels = np.concatenate(all_mask_labels, axis=0)
    return all_mask_points, all_mask_labels

def sample_points_random(masks, n_pos=5, n_neg= 5):
    """
    
        Args: 
            masks: [HW] 
    """
    all_mask_points = []
    all_mask_labels = []
    combined_mask = combine_masks(masks)
    all_pos_ids = np.argwhere(combined_mask == 1)

    for mask in masks:
        assert mask.ndim == 2, 'mask must be 2D'
        pos_indices = np.argwhere(mask == 1)
        neg_indices = np.argwhere(mask == 0)
        pos_replace = True if n_pos > len(pos_indices) else False
        pos_sample_ids = np.random.choice(len(pos_indices), n_pos, replace=pos_replace)
        neg_sample_ids = []
        for j in range(len(n_neg)):
            while True:
                neg_id = np.random.choice(len(neg_indices))
                if neg_id not in all_pos_ids:
                    neg_sample_ids.append(neg_id)
                    break
        neg_sample_ids = np.array(neg_sample_ids)
        
        points = np.concatenate((pos_indices[pos_sample_ids], neg_indices[neg_sample_ids]), axis=0)
        points = np.flip(points, axis=1).copy() # (x, y) sam format
        labels = np.concatenate((np.ones(n_pos), np.zeros(n_neg)), axis=0)
        all_mask_points.append(points)
        all_mask_labels.append(labels)

    all_mask_points = np.concatenate(all_mask_points, axis=0)
    all_mask_labels = np.concatenate(all_mask_labels, axis=0)
    return all_mask_points, all_mask_labels

def combine_masks(masks:list):
    mask = np.zeros_like(masks[0])
    for m in masks:
        mask = np.logical_or(mask, m)
    return mask.astype(np.float32)

################ ################ ################ 
################ visualizations  ################
################ ################ ################ 
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
    parser.add_argument('--output_dir', type=str, default='logs/08-25-2024_21-45-41/predictions')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    image_sets = []
    for idx in range(10):  # Assuming there are 10 sets of images
        image_set = {}
        image_set['Spectrogram'] = plt.imread(output_dir / f"spect_{idx}_raw.png")
        image_set['Mask'] = plt.imread(output_dir / f"spect_{idx}_gt.png")
        image_set['Prediction'] = plt.imread(output_dir / f"spect_{idx}_pred.png")
        image_set['Prompt'] = plt.imread(output_dir / f"spect_{idx}_prompt.png")
        image_sets.append(image_set)

    toggle_visual(image_sets)

