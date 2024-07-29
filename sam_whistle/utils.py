import numpy as np
import matplotlib.pyplot as plt

def get_point_prompts(data, n_pos, n_neg= None, box_pad = 0):
    """sample point prompt from mask.
    Args:
        data: [contours: (n, 2)], [bbox: (4,)]
    """
    assert n_pos or n_neg, 'n_pos or n_neg must be provided'
    spect, bboxes, contours = data
    height, width, _ = spect.shape
    if n_neg is None:
        n_neg = n_pos
    
    prompts = []
    for contour, bbox in zip(contours, bboxes):
        # sample positive points
        n_contour = len(contour)
        stratum_size = n_contour//n_pos
        pos_pts = []
        for i in range(n_pos):
            start_idx = i * stratum_size
            if i == n_pos - 1:
                end_idx = n_contour
            else:
                end_idx = (i + 1) * stratum_size
            
            # Sample one element from the stratum
            stratum = contour[start_idx:end_idx]
            sampled_pair = stratum[np.random.randint(0, len(stratum))]
            pos_pts.append(sampled_pair)

        # sample negative points
        # expanded_bbox
        x_min, y_min, x_max, y_max = bbox
        x_min = x_min - box_pad if x_min - box_pad >0  else 0
        y_min = y_min - box_pad if y_min - box_pad >0  else 0
        x_max = x_max + box_pad if x_max + box_pad < width else width
        y_max = y_max + box_pad if y_max + box_pad < height else height

        sides = []
        perimeter = 0

        # Determine which sides to include based on image constraints
        if x_min > 0:
            sides.append('left')
            perimeter += height
        if y_min > 0:
            sides.append('top')
            perimeter += width
        if x_max < width:
            sides.append('right')
            perimeter += height
        if y_max < height:
            sides.append('bottom')
            perimeter += width


        if perimeter == 0:
            raise ValueError("No sides available for sampling.")
        
        points_per_side = {side: int(n_neg / len(sides)) for side in sides}
        remaining_points = n_neg - sum(points_per_side.values())
        print(points_per_side)
        # Distribute remaining points evenly
        for i in range(remaining_points):
            points_per_side[sides[i % len(sides)]] += 1
        
        # Generate points for each side
        if 'left' in points_per_side:
            pts = np.linspace(y_max, y_min, points_per_side['left'] + 1)
            left_side = [(x_min, y) for y in pts[:-1]]
        if 'top' in points_per_side:
            pts = np.linspace(x_min, x_max, points_per_side['top'] + 1)
            top_side = [(x, y_min) for x in pts[:-1]]
        if 'right' in points_per_side:
            pts = np.linspace(y_min, y_max, points_per_side['right'] + 1)
            right_side = [(x_max, y) for y in pts[:-1]]
        if 'bottom' in points_per_side:
            pts = np.linspace(x_max, x_min, points_per_side['bottom'] + 1)
            bottom_side = [(x, y_max) for x in pts[:-1]]
       
        neg_pts = left_side + bottom_side + right_side + top_side
        # if len(neg_pts) > n_neg:
        #     neg_pts = neg_pts[:n_neg]
        print(len(neg_pts))
        print(np.array(neg_pts))
        points = np.concatenate((pos_pts, neg_pts), axis=0)
        labels = np.concatenate((np.ones(len(pos_pts)), np.zeros(len(neg_pts))), axis=0)
        prompts.append((points, labels))
    return prompts

# visualizations
def show_spect(spect:np.array, fig, save:str = None):
    ax = fig.gca()
    ax.imshow(spect[::-1], origin='lower',cmap='viridis')
    ax.axis('on')
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches='tight')


def show_points(coords, labels, ax, marker_size=50, inverse =None):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], inverse - pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], inverse - neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image[::-1], origin='lower',cmap='viridis')