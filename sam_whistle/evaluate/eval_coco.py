import json
import os
from typing import List

import numpy as np
import pycocotools.mask as maskUtils
from matplotlib import pyplot as plt
from pycocotools.cocoeval import COCOeval
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import clip_by_rect

from sam_whistle.datasets.whistle_coco import WhistleCOCO
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import *

cfg = Config()

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

def polyline_to_polygon(traj: np.ndarray, width: float = 3)-> List[float]:
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
    
    polygon = clip_by_rect(polygon, 0, 0, N_FRAMES, NUM_FREQ_BINS)

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
    rows = NUM_FREQ_BINS - row_top
    rows = np.round(rows - 0.5).astype(int)
    columns = np.round(columns).astype(int)
    coords = np.unique(np.stack([columns, rows], axis=-1), axis=0) # remove duplicate points
    valid_mask  = (coords[:, 0] >= 0) & (coords[:, 0] < N_FRAMES) & (coords[:, 1] >= 0) & (coords[:, 1] < NUM_FREQ_BINS)
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

def get_detections(cfg):
    # TODO: train + test
    whistle_coco_data = os.path.join(cfg.root_dir, 'spec_coco/val/data')
    whistle_coco_label = os.path.join(cfg.root_dir, 'spec_coco/val/labels.json')

    test_set = WhistleCOCO(root=whistle_coco_data, annFile=whistle_coco_label)
    gt_coco = test_set.coco

    stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))['test']
    stems = stems[1:2]
    trackers = {}

    bbox_dts = []
    mask_dts = []

    # First, collect all the detection results
    for stem in stems:
        tracker = TonalTracker(cfg, stem)
        tracker.sam_inference()
        trackers[stem] = tracker
        tracker.build_graph()
        tracker.get_tonals()
        dt_tonals = tracker.dt_tonals
        dt_tonals = [get_dense_annotation(traj) for traj in dt_tonals]
        gt_tonals = tracker.gt_tonals
        gt_tonals = [get_dense_annotation(traj) for traj in gt_tonals]

        stem_img_ids = test_set.audio_to_image[stem]

        for img_id in stem_img_ids:
            data_index = test_set.ids.index(img_id)
            info = test_set[data_index]['info']
            start_frame = info['start_frame']
            # img = test_set[data_index]['img']
            spec_map = tracker.spect_map[::-1, start_frame:start_frame + N_FRAMES]  # lower frequency at the bottom
            dt_trajs = get_segment_annotation(dt_tonals, start_frame)
            gt_trajs = get_segment_annotation(gt_tonals, start_frame)

            for dt_traj in dt_trajs:
            # for dt_traj in gt_trajs:
                traj_pix = tf_to_pix(dt_traj)
                traj_plg = polyline_to_polygon(traj_pix)
                if not traj_plg:
                    continue
                score = get_traj_score(spec_map, traj_pix, tracker.crop_top)
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
                    'score': float(score)  # Ensure score is a float for JSON serialization
                }

                bbox_dts.append(dt_bbox_dict)
                mask_dts.append(dt_mask_dict)
    
    # Evaluate metrics
    for metric in ['bbox', 'segm']:
        coco_dt = gt_coco.loadRes(bbox_dts) if metric == 'bbox' else gt_coco.loadRes(mask_dts)
        coco_eval = COCOeval(gt_coco, coco_dt, metric)
        coco_eval.params.imgIds = stem_img_ids
        coco_eval.params.maxDets = [100, 300, 1000]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


def get_traj_score(spec_map, traj, cut_top):
    """Get the score as the average confidence of the trajectory"""
    traj[:, 1]  = traj[:, 1] - cut_top
    spec_traj = spec_map[traj[:, 1], traj[:, 0]]
    score = spec_traj.mean()
    return score
    

if __name__ == "__main__":
    get_detections(cfg)