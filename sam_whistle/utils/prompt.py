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