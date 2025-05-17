from collections import defaultdict
from copy import deepcopy
import glob
import json
import os
from typing import List
import cv2
import tyro
import argparse
import pickle
from rich import print as rprint
import copy

import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from shapely.geometry import LineString, Point
from shapely.ops import clip_by_rect
from skimage.morphology import skeletonize


from sam_whistle.datasets.whistle_coco import WhistleCOCO
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import *


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

def get_segment_annotation(trajs: List[np.array], start_frame:int):
    """Get the part of annotations within the segment range
    
    Args:
        trajs: list of contours [(time(s), frequency(Hz))]: [(num_points, 2),...]
        start_frame: start frame of the segment
    Returns:
        segment_trajs: list of contours within the segment range [(time(s), frequency(Hz))]: [(num_points, 2),...]
    """
    # determine the range of segments
    start_time = start_frame * HOP_MS / 1000
    end_time = (start_frame + N_FRAMES) * HOP_MS / 1000

    segment_trajs = []
    for traj in trajs:
        if traj[0, 0] <= end_time and traj[-1, 0] >= start_time:
            mask = (traj[:, 0] >= start_time) & (traj[:, 0] <= end_time)
            segment_traj = traj[mask]
            # get the traj whitin range, relative to the segment
            segment_traj[:, 0] = segment_traj[:, 0] - start_time
            segment_trajs.append(segment_traj)
    return segment_trajs

def polyline_to_polygon(traj: np.ndarray, width: float = 3, n_frames =N_FRAMES)-> List[float]:
    """Convert polyline to polygon
    
    Args:
        traj: polyline coordinates [(column, row)]: (num_points, 2) in single spec segment
        width: width of the polyline

    Returns:
        coco segmentation format [x1, y1, x2, y2, ...]
    """
    if len(traj) == 0:
        return False
    if len(traj) == 1:
        # For single point, create a circular polygon
        point = Point(traj[0])
        polygon = point.buffer(width / 2)
    else:
        # Original logic for polyline
        line = LineString(traj)
        polygon = line.buffer(width / 2)

    if polygon.geom_type == "MultiPolygon":
        raise ValueError("The trajectory is too wide, resulting in multiple polygons")
    
    polygon = clip_by_rect(polygon, 0, 0, n_frames, NUM_FREQ_BINS)

    if polygon.is_empty:
            return []
    if polygon.geom_type == "MultiPolygon":
        raise ValueError("The trajectory is too wide, resulting in multiple polygons")

    coco_coords = np.array(polygon.exterior.coords).round(2)
    if len(coco_coords) < 3:
        return []
    return coco_coords.ravel().tolist()  # coco segmentation format [x1, y1, x2, y2, ...]

def tf_to_pix(
    traj: np.ndarray,
    num_freq_bins: int = NUM_FREQ_BINS,
    width: int = N_FRAMES,
):
    """Convert time-frequency coordinates to pixel coordinates within a single spectrogram segment

    Args:
        traj: time-frequency coordinates of the contour [(time(s), frequency(Hz))]: (num_points, 2)

    Returns:
        pixel coordinates of the contour [(column, row)] in int, left bottom origin: (num_points, 2)
    """
    times = traj[:, 0]
    freqs = traj[:, 1]
    columns = times * FRAME_PER_SECOND + 0.5
    row_top = freqs / FREQ_BIN_RESOLUTION
    rows = num_freq_bins - row_top
    rows = np.round(rows - 0.5).astype(int)
    columns = np.round(columns).astype(int)
    coords = np.unique(np.stack([columns, rows], axis=-1), axis=0) # remove duplicate points
    valid_mask  = (coords[:, 0] >= 0) & (coords[:, 0] < width) & (coords[:, 1] >= 0) & (coords[:, 1] < num_freq_bins)
    return coords[valid_mask]

def polygon_to_box(polygon: List[float]):
    """Convert polygon to bounding box
    
    Args:
        polygon: coco segmentation format [x1, y1, x2, y2, ...]
    """
    x = polygon[::2]
    y = polygon[1::2]
    x1, x2 = min(x), max(x)
    y1, y2 = min(y), max(y)
    return [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]

def poly2mask(poly, height, width):
    """Convert polygon to binary mask."""
    if isinstance(poly, list):
        rles = maskUtils.frPyObjects(poly, height, width)
        rle = maskUtils.merge(rles)
    
    mask = maskUtils.decode(rle)
    return mask



def get_detections_record(cfg, model_name, debug=False):
    """Get the detection of each record"""
    stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))
    stems = stems['test'] #+ stems['train']  # test imgs spread over origin train and test audio
    
    if debug:
        # stems = ['Qx-Tt-SCI0608-N1-060814-123433']
        # stems = ['Qx-Dd-SCI0608-N1-060814-150255']
        # stems = ['Qx-Dc-CC0411-TAT11-CH2-041114-154040-s']
        # stems=['Qx-Dc-SC03-TAT09-060516-171606']
        # stems = ['Qx-Dd-SC03-TAT09-060516-211350']
        stems = stems[:1]

    # trackers = {}
    img_to_whistles = dict()
    scores_all = []

    for i, stem in enumerate(stems):
        tracker = TonalTracker(cfg, stem)
        if model_name == 'sam':
            tracker.sam_inference()
        elif model_name == 'sam2':
            tracker.sam2_inference()
        elif model_name == 'dw':
            tracker.dw_inference()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        # trackers[stem] = tracker
        tracker.build_graph()
        tracker.get_tonals()
        dt_tonals = tracker.dt_tonals
        dt_tonals = [get_dense_annotation(traj) for traj in dt_tonals]
        rprint(f"stem: {stem}, dt_tonals: {len(dt_tonals)}")
        gt_tonals = tracker.gt_tonals
        gt_tonals = [get_dense_annotation(traj) for traj in gt_tonals]
        rprint(f"stem: {stem}, gt_tonals: {len(gt_tonals)}")

        conf_map =tracker.conf_map
        width = conf_map.shape[1]

        def unique_pix(traj):
            unique_x = np.unique(traj[:, 0])
            averaged_y = np.zeros_like(unique_x)
            for i, x in enumerate(unique_x):
                y_values = traj[traj[:, 0] == x][:, 1]
                averaged_y[i] = int(np.round(np.mean(y_values)))
            unique_traj = np.column_stack((unique_x, averaged_y))
            return unique_traj
        
        tonal_dts = []
        tonal_gts = []
        bounds_gt = []
        plgs = []
        scores = []
        for i, gt_traj in enumerate(gt_tonals):
            traj_pix = tf_to_pix(gt_traj, width=tracker.W)
            traj_pix = unique_pix(traj_pix)
            try:
                _, bound = get_traj_valid((tracker.raw_spect*255).numpy().astype(np.uint8), traj_pix)
            except:
                import pdb; pdb.set_trace()
            tonal_gts.append(traj_pix)
            bounds_gt.append(bound)

            gt_traj_plg = polyline_to_polygon(traj_pix, n_frames=width)
            plgs.append(gt_traj_plg)
            

        # if debug:
        #     dt_tonals = deepcopy(gt_tonals)
        #     conf_map = poly2mask(plgs, *conf_map.shape)

        for dt_traj in dt_tonals:
            traj_pix = tf_to_pix(dt_traj, width=width)
            # score = get_traj_score(conf_map, deepcopy(traj_pix))
            # scores.append(score)
            traj_pix = unique_pix(traj_pix)
            tonal_dts.append(traj_pix)

        rprint(f'stem:{stem}, scores: {np.mean(scores):.4f}, std: {np.std(scores):.4f}, min: {np.min(scores):.4f}, max: {np.max(scores):.4f}') if len(scores) > 0 else None
        img_to_whistles[stem] = {
            'gts': tonal_gts,
            # 'boudns_gt': bounds_gt,
            'dts': tonal_dts,
            # 'scores': np.mean(scores).item(),
            'img_id': stem,
            'w': width,
        }
        scores_all.extend(scores)
        del tracker
    
    rprint(f'ALL scores: {np.mean(scores_all):.4f}, std: {np.std(scores_all):.4f}, min: {np.min(scores_all):.4f}, max: {np.max(scores_all):.4f}') if len(scores_all) > 0 else None
    sum_gts = sum([len(whistles['gts']) for whistles in img_to_whistles.values()])
    sum_dts = sum([len(whistles['dts']) for whistles in img_to_whistles.values()])
    rprint(f'gathered gts: {sum_gts}, dts: {sum_dts}')
    return img_to_whistles


def get_detections_coco(cfg, model_name='sam', debug=False):
    whistle_coco_data = os.path.join(cfg.root_dir, 'coco/val/data')
    whistle_coco_label = os.path.join(cfg.root_dir, 'coco/val/labels.json')

    test_set = WhistleCOCO(root=whistle_coco_data, annFile=whistle_coco_label)
    gt_coco = test_set.coco

    stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))
    stems = stems['test'] #+ stems['train']  # test imgs spread over origin train and test audio
    
    
    if debug:
        # stems = ['Qx-Tt-SCI0608-N1-060814-123433']
        # stems = ['Qx-Dd-SCI0608-N1-060814-150255']
        stems = stems[:1]
    # trackers = {}

    bbox_dts = []
    mask_dts = []
    # mask_gts = []
    scores_all = []

    gt_image_ids = []
    # First, collect all the detection results
    for stem in stems:
        scores = []
        tracker = TonalTracker(cfg, stem)
        if model_name == 'sam':
            tracker.sam_inference()
        elif model_name == 'sam2':
            tracker.sam2_inference()
        elif model_name == 'dw':
            tracker.dw_inference()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        # trackers[stem] = tracker
        tracker.build_graph()
        tracker.get_tonals()
        dt_tonals = tracker.dt_tonals
        dt_tonals = [get_dense_annotation(traj) for traj in dt_tonals]
        rprint(f"stem: {stem}, dt_tonals: {len(dt_tonals)}")
        gt_tonals = tracker.gt_tonals
        gt_tonals = [get_dense_annotation(traj) for traj in gt_tonals]
        rprint(f"stem: {stem}, gt_tonals: {len(gt_tonals)}")

        stem_img_ids = test_set.audio_to_image[stem]
        gt_image_ids.extend(stem_img_ids)

        for img_id in stem_img_ids:
            data_index = test_set.ids.index(img_id)
            info = test_set[data_index]['info']
            start_frame = info['start_frame']
            # img = test_set[data_index]['img']
            # conf_map = tracker.spect_map[::-1, start_frame:start_frame + N_FRAMES]  # lower frequency at the bottom
            conf_map = tracker.conf_map[:, start_frame:start_frame + N_FRAMES]
            dt_trajs = get_segment_annotation(dt_tonals, start_frame)
            # Note: num does not match the anno file, some are in train, some in test
            gt_trajs = get_segment_annotation(gt_tonals, start_frame)
            
            if debug:
                dt_trajs = deepcopy(gt_trajs)
                plgs = []
                for gt_traj in gt_trajs:
                    gt_traj_pix = tf_to_pix(gt_traj)
                    gt_traj_plg = polyline_to_polygon(gt_traj_pix)
                    plgs.append(gt_traj_plg)
                conf_map = poly2mask(plgs, NUM_FREQ_BINS, N_FRAMES)

            if len(dt_trajs) < 1:
                continue

            for dt_traj in dt_trajs:
                traj_pix = tf_to_pix(dt_traj)  # pixel coordinates in 769x1500 low freq at the bottom
                traj_plg = polyline_to_polygon(traj_pix)
                if not traj_plg:
                    continue

                score = get_traj_score(conf_map, deepcopy(traj_pix))
                scores.append(score)
                bbox = polygon_to_box(traj_plg)
                dt_bbox_dict = {
                    'image_id': img_id,
                    'bbox': bbox,
                    'score': float(score),  # Ensure score is a float for JSON serialization
                    'category_id': 1
                }

                rles = maskUtils.frPyObjects([traj_plg], NUM_FREQ_BINS, N_FRAMES)
                rle = maskUtils.merge(rles)

                dt_mask_dict = {
                    'image_id': img_id,
                    'segmentation': rle,
                    'category_id': 1,
                    'score': float(score),  # Ensure score is a float for JSON serialization
                    'traj_pix': traj_pix,
                }

                bbox_dts.append(dt_bbox_dict)
                mask_dts.append(dt_mask_dict)
        rprint(f'stem:{stem}, scores: {np.mean(scores):.4f}, std: {np.std(scores):.4f}, min: {np.min(scores):.4f}, max: {np.max(scores):.4f}')
        scores_all.extend(scores)
        del tracker
        
    rprint(f'ALL scores: {np.mean(scores):.4f}, std: {np.std(scores):.4f}, min: {np.min(scores):.4f}, max: {np.max(scores):.4f}')
    # Evaluate metrics
    bbox_coco = gt_coco.loadRes(bbox_dts)
    mask_coco = gt_coco.loadRes(mask_dts)
    # gt_coco = gt_coco.loadRes(mask_gts)
    
    if debug:
        def check_areas(coco):
            small_num = 0
            large_num = 0
            medium_num = 0
            for ann in coco.anns.values():
                area = ann['area']
                if area < 32**2:
                    small_num += 1
                elif area > 96**2:
                    large_num += 1
                else:
                    medium_num += 1
            rprint(f"small area: {small_num},  medium area: {medium_num}, large area: {large_num},")
        check_areas(bbox_coco)
        check_areas(mask_coco)

    for metric in ['bbox', 'segm']:
        coco_eval = COCOeval(gt_coco, bbox_coco, metric) if metric == 'bbox' else COCOeval(gt_coco, mask_coco, metric)
        # coco_eval.params.imgIds = stem_img_ids
        coco_eval.params.imgIds = gt_image_ids  # use detected subsets
        coco_eval.params.maxDets = [100, 300, 1000]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return bbox_coco, mask_coco, gt_coco


def get_traj_score(conf_map, traj):
    """Get the score as the average confidence of the trajectory"""
    # traj[:, 1]  = traj[:, 1] - cut_top
    conf_traj = conf_map[traj[:, 1], traj[:, 0]]
    score = conf_traj.mean()
    return score

def get_traj_valid(conf_map, traj):

    conf_traj = conf_map[traj[:, 1], traj[:, 0]]
    score = conf_traj.mean()
    sorted_idx = np.argsort(conf_traj)
    ratio = 0.3
    if len(sorted_idx) > 0:
        bound = conf_traj[sorted_idx[int(len(sorted_idx) * ratio)]]
        return score, bound
    else:
        raise ValueError("No valid points in the trajectory")


def bresenham_line(p1, p2):
    """
    Implements Bresenham's line algorithm for grid-aligned paths.
    This creates the most direct path between two points while staying on the grid.
    
    Args:
        p1: Starting point (x1, y1)
        p2: Ending point (x2, y2)
        
    Returns:
        List of points [(x1,y1), ..., (x2,y2)] forming a continuous path
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Initialize the path with the starting point
    path = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        path.append((x1, y1))
        
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return path


def midpoint_interpolation(p1, p2):
    """
    Recursively creates a path by adding midpoints between points.
    
    Args:
        p1: Starting point (x1, y1)
        p2: Ending point (x2, y2)
        
    Returns:
        List of points forming a continuous path
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Base case: points are adjacent or the same
    if abs(x2 - x1) <= 1 and abs(y2 - y1) <= 1:
        return [p1, p2]
    
    # Find the midpoint (rounded to integers)
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    midpoint = (mid_x, mid_y)
    
    # Recursively find paths for each half
    first_half = midpoint_interpolation(p1, midpoint)
    second_half = midpoint_interpolation(midpoint, p2)
    
    # Combine the paths (avoid duplicating the midpoint)
    return first_half[:-1] + second_half


def simple_weighted_path(points, i, window=2):
    """
    A simpler weighted path method that doesn't use BFS.
    Uses direct interpolation with local direction awareness.
    
    Args:
        points: All trajectory points
        i: Current index
        window: Number of points to consider on each side
        
    Returns:
        List of points forming a continuous path
    """
    # Ensure all points are tuples
    points = [tuple(p) for p in points]
    current = points[i]
    next_point = points[i + 1]
    
    # If points are close, use Bresenham's algorithm
    if abs(next_point[0] - current[0]) <= 3 and abs(next_point[1] - current[1]) <= 3:
        return bresenham_line(current, next_point)
    
    # Get neighboring points within window
    start_idx = max(0, i - window)
    end_idx = min(len(points) - 1, i + 1 + window)
    neighbors = points[start_idx:end_idx + 1]
    
    # Simple weighted path based on direction awareness
    if len(neighbors) <= 2:
        return bresenham_line(current, next_point)
    
    # Calculate primary direction from neighbors
    x_coords = [p[0] for p in neighbors]
    y_coords = [p[1] for p in neighbors]
    
    # Simple linear trend (direction)
    if len(neighbors) >= 3:
        x_diffs = [x_coords[j+1] - x_coords[j] for j in range(len(x_coords)-1)]
        y_diffs = [y_coords[j+1] - y_coords[j] for j in range(len(y_coords)-1)]
        avg_x_diff = sum(x_diffs) / len(x_diffs)
        avg_y_diff = sum(y_diffs) / len(y_diffs)
    else:
        avg_x_diff = next_point[0] - current[0]
        avg_y_diff = next_point[1] - current[1]
    
    # Create a path with awareness of the typical step size
    path = [current]
    
    # Current position
    cx, cy = current
    tx, ty = next_point
    
    # Step sizes (make them integers between 1-3 based on average direction)
    step_x = max(1, min(3, int(abs(avg_x_diff)) or 1)) * (1 if tx > cx else -1 if tx < cx else 0)
    step_y = max(1, min(3, int(abs(avg_y_diff)) or 1)) * (1 if ty > cy else -1 if ty < cy else 0)
    
    # We'll adjust step_x and step_y to ensure we don't overshoot
    while (cx, cy) != next_point:
        # Decide which direction to move
        if abs(cx - tx) > abs(cy - ty):
            # Move in x direction
            new_cx = cx + step_x
            # Check if we'd overshoot
            if (step_x > 0 and new_cx > tx) or (step_x < 0 and new_cx < tx):
                new_cx = tx
            cx = new_cx
        else:
            # Move in y direction
            new_cy = cy + step_y
            # Check if we'd overshoot
            if (step_y > 0 and new_cy > ty) or (step_y < 0 and new_cy < ty):
                new_cy = ty
            cy = new_cy
        
        # Add new point to path
        new_point = (cx, cy)
        
        # Check if we need to fill gaps (ensure path is grid-continuous)
        last_point = path[-1]
        if abs(new_point[0] - last_point[0]) > 1 or abs(new_point[1] - last_point[1]) > 1:
            # Fill gap with Bresenham
            gap_filler = bresenham_line(last_point, new_point)
            path.extend(gap_filler[1:])
        else:
            path.append(new_point)
    
    return path


def bezier_grid_path(points, start_idx, end_idx, steps=10):
    """
    Creates a Bezier curve between points and snaps it to grid.
    
    Args:
        points: All trajectory points
        start_idx: Index of starting point
        end_idx: Index of ending point
        steps: Number of interpolation steps
        
    Returns:
        List of grid points approximating a Bezier curve
    """
    # Extract points
    p0 = points[start_idx]
    p3 = points[end_idx]
    
    # Use neighboring points to determine control points if available
    if start_idx > 0 and end_idx < len(points) - 1:
        # Control points based on neighboring points
        prev = points[start_idx - 1]
        next_point = points[end_idx + 1]
        
        # Create control points by extending the lines from neighbors
        dx1 = p0[0] - prev[0]
        dy1 = p0[1] - prev[1]
        p1 = (p0[0] + dx1 // 2, p0[1] + dy1 // 2)
        
        dx2 = p3[0] - next_point[0]
        dy2 = p3[1] - next_point[1]
        p2 = (p3[0] + dx2 // 2, p3[1] + dy2 // 2)
    else:
        # Default control points for endpoints
        dx = p3[0] - p0[0]
        dy = p3[1] - p0[1]
        p1 = (p0[0] + dx // 3, p0[1] + dy // 3)
        p2 = (p0[0] + 2 * dx // 3, p0[1] + 2 * dy // 3)
    
    # Generate Bezier curve points
    curve_points = []
    for t in np.linspace(0, 1, steps):
        # Cubic Bezier formula
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        
        # Round to nearest grid point
        curve_points.append((round(x), round(y)))
    
    # Ensure the path is continuous by filling any gaps
    grid_path = [p0]
    for i in range(1, len(curve_points)):
        prev = grid_path[-1]
        current = curve_points[i]
        
        # If points aren't adjacent, fill the gap with Bresenham
        if abs(current[0] - prev[0]) > 1 or abs(current[1] - prev[1]) > 1:
            connecting_points = bresenham_line(prev, current)
            grid_path.extend(connecting_points[1:])
        else:
            grid_path.append(current)
    
    # Ensure end point is included
    if grid_path[-1] != p3:
        connecting_points = bresenham_line(grid_path[-1], p3)
        grid_path.extend(connecting_points[1:])
    
    return grid_path

def mask_to_whistle(mask, method='bresenham'):
    """convert the instance mask to whistle contour, use skeleton methods
    
    Args
        mask: instance mask (H, W)
    Return
        whistle: (N,2) in pixel coordinates
    """
    mask = mask.astype(np.uint8)
    skeleton = skeletonize(mask).astype(np.uint8)
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[:, [0, -1]] = True
    border_pixels = mask & border_mask
    skeleton = skeleton | border_pixels
    whistle = np.array(np.nonzero(skeleton)).T # [N x (y, x)]
    whistle = np.flip(whistle, axis=1)  # [(x, y)]
    whistle = np.unique(whistle, axis=0)  # remove duplicate points
    whistle = whistle[whistle[:, 0].argsort()]
    assert whistle.ndim ==2 and whistle.shape[1] == 2, f"whistle shape: {whistle.shape}"

    # connect fragmented whistle points
    whistle = whistle.tolist()
    whistle_ =[whistle[0]]
    for i in range(len(whistle) - 1):
        current = whistle[i]
        next_point = whistle[i + 1]
        
        # Generate intermediate points between current and next_point
        if method == 'bresenham':
            intermediate_points = bresenham_line(current, next_point)
        elif method == 'midpoint':
            intermediate_points = midpoint_interpolation(current, next_point)
        elif method == 'bezier':
            intermediate_points = bezier_grid_path(whistle, i, i+1)
        elif method == 'weighted':
            intermediate_points = simple_weighted_path(whistle, i)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add intermediate points to trajectory (skip the first one as it's already included)
        whistle_.extend(intermediate_points[1:])
    
    # Group by x-coordinate and select one y-value per x
    whistle_ = np.array(whistle_)
    unique_x = np.unique(whistle_[:, 0])
    averaged_y = np.zeros_like(unique_x)
    for i, x in enumerate(unique_x):
        y_values = whistle_[whistle_[:, 0] == x][:, 1]
        averaged_y[i] = int(np.round(np.mean(y_values)))
    unique_whistle = np.column_stack((unique_x, averaged_y))

    return unique_whistle

def gather_whistles(coco_gt:COCO, coco_dt:COCO, filter_dt=0, valid_gt=False, root_dir=None, debug=False):
    """gather per image whistles from instance masks"""
    
    if debug:
        coco_dt = deepcopy(coco_gt)
        for ann in coco_dt.anns.values():
            ann['score'] = 1.0

    img_to_whistles = dict()
    for img_id in coco_dt.imgs.keys():
        img = coco_dt.imgs[img_id]
        h, w = img['height'], img['width']
        dt_anns = coco_dt.imgToAnns[img_id]
        if debug and len(dt_anns) == 0:  # keep gt
            continue
        gt_anns = coco_gt.imgToAnns[img_id]
        gt_masks = [coco_gt.annToMask(ann) for ann in gt_anns]

        dt_masks = []
        for i, ann in enumerate(dt_anns):
            score = ann['score']
            mask = coco_dt.annToMask(ann)
            if score > filter_dt:
                dt_masks.append(mask)
            else:
                continue

        gt_whistles = [mask_to_whistle(mask) for mask in gt_masks]
        
        # bounds = []
        # gt_whistles_ = []
        # img_info = coco_gt.imgs[img_id]
        # image = cv2.imread(os.path.join(root_dir,'coco/val/data' , img_info['file_name']))
        # for whistle in gt_whistles:
        #     whistle = np.array(whistle)
        #     score, bound = get_traj_valid(image[...,0], whistle)
        #     bounds.append(bound)
        #     gt_whistles_.append(whistle)
        # gt_whistles = gt_whistles_

        dt_whistles = [mask_to_whistle(mask) for mask in dt_masks if mask.sum()>0]
        
        if debug:
            assert len(gt_whistles) == len(dt_whistles), f"gt and dt should have the \
            same number of whistles, gt: {len(gt_whistles)}, dt: {len(dt_whistles)}"
        
        img_to_whistles[img['id']] = {
            'gts': gt_whistles,
            # 'boudns_gt': bounds,
            # 'boudns_gt': None,
            'dts': dt_whistles,
            'w': w,
            'img_id': img_id,
        }
    sum_gts = sum([len(whistles['gts']) for whistles in img_to_whistles.values()])
    sum_dts = sum([len(whistles['dts']) for whistles in img_to_whistles.values()])
    rprint(f'gathered {len(img_to_whistles)} images with {sum_gts} gt whistles, {sum_dts} dt whistles')
    return img_to_whistles


def compare_whistles(gts, dts, w, img_id, boudns_gt=None, valid_gt = False, debug=False):
    """given whistle gt and dt in evaluation unit and get comparison results
    Args:
        gts, dts: N, 2 in format of y, x(or t, f)
    """
    gt_num = len(gts)
    # boudns_gt = np.array(boudns_gt)
    gt_ranges = np.zeros((gt_num, 2))
    gt_durations = np.zeros(gt_num)

    dt_num = len(dts)
    dt_ranges = np.zeros((dt_num, 2))
    dt_durations = np.zeros(dt_num)
    
    if debug:
        # for i in range(dt_num):
        #    assert (gts[i]== dts[i]).all(), f"gt and dt should not be the same {len(gts)} vs {len(dts)}"
        pass

    for gt_idx, gt in enumerate(gts):
        gt_start_x = max(0, gt[:, 0].min())
        gt_end_x = min(w -1 , gt[:, 0].max())
        gt_dura = gt_end_x + 1 - gt_start_x  # add 1 in pixel
        gt_durations[gt_idx] = gt_dura
        gt_ranges[gt_idx] = (gt_start_x, gt_end_x)
  
    for dt_idx, dt in enumerate(dts):
        dt_start_x = max(0, dt[:, 0].min())
        dt_end_x = min(w, dt[:, 0].max())
        dt_ranges[dt_idx] = (dt_start_x, dt_end_x)
        dt_durations[dt_idx] = dt_end_x + 1 - dt_start_x # add 1 in pixel
    
    dt_false_pos_all = list(range(dt_num))
    dt_true_pos_all = []
    dt_true_pos_valid = []
    gt_matched_all = []
    gt_matched_valid = []
    gt_missed_all = []
    gt_missed_valid = []
    all_deviation = []
    all_covered = []
    all_dura = []

    # go through each ground truth
    for gt_idx, gt in enumerate(gts):
        gt_start_x, gt_end_x = gt_ranges[gt_idx]
        gt_dura = gt_durations[gt_idx]

        # if gt_dura < 2:
        #     continue

        # Note: remove interpolation and snr validation that filter low gt snr level
        # which requires addition input
        dt_start_xs, dt_end_xs = dt_ranges[:, 0], dt_ranges[:, 1]
        ovlp_cond = (dt_start_xs <= gt_start_x) & (dt_end_xs >= gt_start_x) \
                        | (dt_start_xs>= gt_start_x) & (dt_start_xs <= gt_end_x)
        ovlp_dt_ids = np.nonzero(ovlp_cond)[0]

        matched = False
        deviations = []
        covered = 0
        # cnt = 0
        # dt_matched = []
        # dt_matched_dev = []
        # Note remove duration filter short < 75 pix whistle

        valid = True
        if valid_gt:
            if gt_dura < 75: # or boudns_gt[gt_idx] < 3:
                valid= False
        # rprint(f'valid:{valid}, gt_dura: {gt_dura}, boudns_gt: {boudns_gt[gt_idx]}')


        for ovlp_dt_idx in ovlp_dt_ids:
            ovlp_dt = dts[ovlp_dt_idx]
            dt_xs, dt_ys = ovlp_dt[:, 0], ovlp_dt[:, 1]
            dt_ovlp_x_idx = np.nonzero((dt_xs >= gt_start_x) & (dt_xs <= gt_end_x))[0]
            dt_ovlp_xs = dt_xs[dt_ovlp_x_idx]
            dt_ovlp_ys = dt_ys[dt_ovlp_x_idx]

            # Note: remove interpolation
            gt_ovlp_ys = gt[:, 1][np.searchsorted(gt[:, 0], dt_ovlp_xs)]
            deviation = np.abs(gt_ovlp_ys - dt_ovlp_ys)
            deviation_tolerence = 350 / 125
            if debug:
                # deviation_tolerence = 0.1
                pass
            if len(deviation)> 0 and np.mean(deviation) <= deviation_tolerence:
                matched = True
                
                # cnt += 1
                # dt_matched.append(ovlp_dt_idx)
                if ovlp_dt_idx in dt_false_pos_all:
                    dt_false_pos_all.remove(ovlp_dt_idx)
                # TODO: has multiplications
                dt_true_pos_all.append(ovlp_dt_idx)  
                # TODO: the deviation and coverage of same overlap can be counted multiple times
                deviations.extend(deviation)

                # dt_matched_dev.append(deviation)
                
                covered += dt_ovlp_xs.max() - dt_ovlp_xs.min() + 1
                if valid:
                    dt_true_pos_valid.append(ovlp_dt_idx)


        # if debug and cnt > 1:
        #     rprint(f"img_id: {img_id}, multiplication gt_id: {gt_idx} cnt: {cnt}")
        #     rprint(f"gt: {gt.T}",)
        #     for i, idx in enumerate(dt_matched):
        #         rprint(f"dt_idx: {idx}")
        #         rprint(f"dt: {dts[idx].T}")
        #         print(f"deviation: {dt_matched_dev[i].mean()}")
        
        if matched:
            gt_matched_all.append(gt_idx)
            if valid:
                gt_matched_valid.append(gt_idx)
        else:
            gt_missed_all.append(gt_idx)
            if valid:
                gt_missed_valid.append(gt_idx)

        if matched:
            gt_deviation = np.mean(deviations)
            all_deviation.append(gt_deviation) 
            all_covered.append(covered)
            all_dura.append(gt_dura)

    if debug:
        # if dt_false_pos_all:
        #     for dt_fp_idx in dt_false_pos_all:
        #         rprint(f'img_id: {img_id}, dt_id:{dt_fp_idx}, fp_num: {len(dt_false_pos_all)}')
        #         rprint(f'dt: {dts[dt_fp_idx]}')
        #         # rprint(f'gt: {gts[dt_fp_idx]}')
        pass
                
    res = {
        # TODO TP and FN are calculated based on dt and gt respectively
        'dt_false_pos_all': len(dt_false_pos_all),
        'dt_true_pos_all': len(dt_true_pos_all),
        'gt_matched_all': len(gt_matched_all),
        'gt_missed_all': len(gt_missed_all),
        'dt_true_pos_valid': len(dt_true_pos_valid),
        'gt_matched_valid': len(gt_matched_valid),
        'gt_missed_valid': len(gt_missed_valid),
        'all_deviation': all_deviation,
        'all_covered': all_covered,
        'all_dura': all_dura
    }
    return res


def accumulate_wistle_results(img_to_whistles,valid_gt, debug=False):
    """accumulate the whistle results for all images (segment or entire audio)"""
    accumulated_res = {
        'dt_false_pos_all': 0,
        'dt_true_pos_all': 0,
        'gt_matched_all': 0,
        'gt_missed_all': 0,
        'dt_true_pos_valid': 0,
        'gt_matched_valid': 0,
        'gt_missed_valid': 0,
        'all_deviation': [],
        'all_covered': [],
        'all_dura': []
    }
    for img_id, whistles in img_to_whistles.items():
        res = compare_whistles(**whistles, valid_gt = valid_gt, debug=debug)
        # rprint(f'img_id: {img_id}')
        # rprint(sumerize_whisle_results(res))
        accumulated_res['dt_false_pos_all'] += res['dt_false_pos_all']
        accumulated_res['dt_true_pos_all'] += res['dt_true_pos_all']
        accumulated_res['dt_true_pos_valid'] += res['dt_true_pos_valid']
        accumulated_res['gt_matched_all'] += res['gt_matched_all']
        accumulated_res['gt_matched_valid'] += res['gt_matched_valid']
        accumulated_res['gt_missed_all'] += res['gt_missed_all']
        accumulated_res['gt_missed_valid'] += res['gt_missed_valid']
        accumulated_res['all_deviation'].extend(res['all_deviation'])
        accumulated_res['all_covered'].extend(res['all_covered'])
        accumulated_res['all_dura'].extend(res['all_dura'])
    return accumulated_res

def sumerize_whisle_results(accumulated_res):
    """sumerize the whistle results"""
    accumulated_res = copy.deepcopy(accumulated_res)
    dt_fp = accumulated_res['dt_false_pos_all']
    dt_tp = accumulated_res['dt_true_pos_all']
    dt_tp_valid = accumulated_res['dt_true_pos_valid']
    gt_tp = accumulated_res['gt_matched_all']
    gt_tp_valid = accumulated_res['gt_matched_valid']
    gt_fn = accumulated_res['gt_missed_all']
    gt_fn_valid = accumulated_res['gt_missed_valid']

    precision = dt_tp / (dt_tp + dt_fp) if (dt_tp + dt_fp) > 0 else 0
    precision_valid = dt_tp_valid / (dt_tp_valid + dt_fp) if (dt_tp_valid + dt_fp) > 0 else 0
    recall = gt_tp / (gt_tp + gt_fn) if (gt_tp + gt_fn) > 0 else 0
    recall_valid = gt_tp_valid / (gt_tp_valid + gt_fn_valid) if (gt_tp_valid + gt_fn_valid) > 0 else 0
    frag = dt_tp / gt_tp if gt_tp > 0 else 0
    frag_valid = dt_tp_valid / gt_tp_valid if gt_tp_valid > 0 else 0

    accumulated_res['all_deviation'] = np.mean(accumulated_res['all_deviation']).item()
    accumulated_res['all_covered'] = np.sum(accumulated_res['all_covered']).item()
    accumulated_res['all_dura'] = np.sum(accumulated_res['all_dura']).item()
    coverage = accumulated_res['all_covered'] / accumulated_res['all_dura'] if accumulated_res['all_dura'] > 0 else 0

    summary = {
        'precision': precision,
        'recall': recall,
        'frag': frag,
        'coverage': coverage,
        'gt_n':(gt_tp_valid + gt_fn_valid),
        'dt_n':(dt_tp_valid + dt_fp),
        'precision_valid': precision_valid,
        'recall_valid': recall_valid,
        'frag_valid': frag_valid,
    }
    return summary


if __name__ == "__main__":
    # classes = ['bottlenose', 'common', 'melon-headed','spinner']
    # meta = defaultdict(list)
    # for s in classes[:2]:
    #     root_dir = os.path.join(os.path.expanduser("~"),f'storage/DCLDE/whale_whistle/{s}')
    #     bin_files = glob.glob('*.bin', root_dir=root_dir)
    #     stems = []
    #     for bin in bin_files:
    #         gts = utils.load_annotation(os.path.join(root_dir, bin))
    #         if gts:
    #             stems.append(bin.replace('.bin', ''))
    #     meta['test'].extend(stems)
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

    parser = argparse.ArgumentParser(description="Evaluate SAM on COCO dataset")
    parser.add_argument("--model_name", type=str, default="sam")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--type", type=str, default="record")
    known_args, unknown_args = parser.parse_known_args()

    cfg = tyro.cli(Config, args=unknown_args)
    # known_args.debug = True
    # cfg.thre_norm = 0.06939393939393938
    # cfg.thre_norm = 0.05
    if known_args.type == 'coco':
        bbox_dts, mask_dts, gt_coco = get_detections_coco(cfg, model_name=known_args.model_name, debug=known_args.debug)
        
        data_name = cfg.root_dir.split('/')[-1]

        with open(f'outputs/dt_result_{data_name}_{known_args.model_name}{'_debug' if known_args.debug else ''}.pkl', 'wb') as f:
            pickle.dump({'bbox_coco': bbox_dts, 'mask_coco': mask_dts, 'gt_coco': gt_coco}, f)

        with open(f'outputs/dt_result_{data_name}_{known_args.model_name}{'_debug' if known_args.debug else ''}.pkl', 'rb') as f:
            dt_results = pickle.load(f)
            gt_coco = dt_results['gt_coco']
            mask_dts = dt_results['mask_coco']

        img_to_whistles = gather_whistles(gt_coco, mask_dts, root_dir=cfg.root_dir, debug=known_args.debug)
    elif known_args.type == 'record':
        img_to_whistles = get_detections_record(cfg, model_name=known_args.model_name, debug=known_args.debug)
    elif known_args.type == 'coco_record':
        pass
    else:
        raise ValueError(f"Unknown type: {known_args.type}")
 

    res = accumulate_wistle_results(img_to_whistles, valid_gt=True, debug=known_args.debug)
    summary = sumerize_whisle_results(res)
    rprint(summary)

    

