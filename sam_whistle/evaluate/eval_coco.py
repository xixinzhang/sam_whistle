from collections import defaultdict
from copy import deepcopy
import json
import os
from typing import List
import tyro
import argparse
import pickle
from rich import print as rprint

import numpy as np
import pycocotools.mask as maskUtils
from matplotlib import pyplot as plt
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from shapely.geometry import LineString, Point, Polygon
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

def get_detections(cfg, model_name='sam', debug=False):
    whistle_coco_data = os.path.join(cfg.root_dir, 'spec_coco/val/data')
    whistle_coco_label = os.path.join(cfg.root_dir, 'spec_coco/val/labels.json')

    test_set = WhistleCOCO(root=whistle_coco_data, annFile=whistle_coco_label)
    gt_coco = test_set.coco

    stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))
    stems = stems['test'] + stems['train']  # test imgs spread over origin train and test audio
    if debug:
        # stems = ['Qx-Tt-SCI0608-N1-060814-123433']
        # stems = ['Qx-Dd-SCI0608-N1-060814-150255']
        stems = stems[:1]
    trackers = {}

    bbox_dts = []
    mask_dts = []

    gt_image_ids = []
    # First, collect all the detection results
    for stem in stems:
        tracker = TonalTracker(cfg, stem)
        if model_name == 'sam':
            tracker.sam_inference()
        elif model_name == 'sam2':
            tracker.sam2_inference()
        elif model_name == 'dw':
            tracker.dw_inference()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        trackers[stem] = tracker
        tracker.build_graph()
        tracker.get_tonals()
        dt_tonals = tracker.dt_tonals
        dt_tonals = [get_dense_annotation(traj) for traj in dt_tonals]
        if debug:
            gt_tonals = tracker.gt_tonals
            gt_tonals = [get_dense_annotation(traj) for traj in gt_tonals]
            rprint(f"stem: {stem}, dt_tonals: {len(dt_tonals)}")
            rprint(f"stem: {stem}, gt_tonals: {len(gt_tonals)}")

        stem_img_ids = test_set.audio_to_image[stem]
        gt_image_ids.extend(stem_img_ids)

        for img_id in stem_img_ids:
            data_index = test_set.ids.index(img_id)
            info = test_set[data_index]['info']
            start_frame = info['start_frame']
            # img = test_set[data_index]['img']
            spec_map = tracker.spect_map[::-1, start_frame:start_frame + N_FRAMES]  # lower frequency at the bottom
            dt_trajs = get_segment_annotation(dt_tonals, start_frame)
            # Note: num does not match the anno file, some are in train, some in test
            
            if debug:
                gt_trajs = get_segment_annotation(gt_tonals, start_frame)
                dt_trajs = deepcopy(gt_trajs)

            for dt_traj in dt_trajs:
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
    bbox_coco = gt_coco.loadRes(bbox_dts)
    mask_coco = gt_coco.loadRes(mask_dts)
    
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


def get_traj_score(spec_map, traj, cut_top):
    """Get the score as the average confidence of the trajectory"""
    traj[:, 1]  = traj[:, 1] - cut_top
    spec_traj = spec_map[traj[:, 1], traj[:, 0]]
    score = spec_traj.mean()
    return score


def mask_to_whistle(mask):
    """convert the instance mask to whistle contour, use skeleton methods
    
    Args
        mask: instance mask (H, W)
    Return
        whistle: (N,2) in pixel coordinates
    """
    mask = mask.astype(np.uint8)
    skeleton = skeletonize(mask).astype(np.uint8)
    whistle = np.array(np.nonzero(skeleton)).T # [(y, x]
    whistle = np.flip(whistle, axis=1)  # [(x, y)]

    # Group by x-coordinate and select one y-value per x
    x_to_y = defaultdict(list)
    for x, y in whistle:
        x_to_y[x].append(y)
    unique_whistle = np.array([(x, int(np.round(np.mean(y)))) for x, y in x_to_y.items()])
    unique_whistle = unique_whistle[np.argsort(unique_whistle[:, 0])]

    num_x = len(np.unique(unique_whistle[:, 0]))
    if num_x != len(unique_whistle):
        rprint(f"num_x: {(num_x, len(unique_whistle))}")
    

    whisle_x = unique_whistle[:, 0]
    for i in range(whisle_x[0], whisle_x[-1]+1):
        if i not in unique_whistle[:, 0]:
            rprint(f"has missing x: {i}")
            break

    return unique_whistle


def gather_whistles(coco_gt:COCO, coco_dt:COCO, debug=False):
    """gather per image whistles from instance masks"""
    
    if debug:
        coco_dt = deepcopy(coco_gt)
    img_to_whistles = dict()
    for img_id in coco_dt.imgs.keys():
        img = coco_dt.imgs[img_id]
        h, w = img['height'], img['width']
        dt_anns = coco_dt.imgToAnns[img_id]
        if debug and len(dt_anns) == 0:  # keep gt
            continue
        gt_anns = coco_gt.imgToAnns[img_id]
        gt_masks = [coco_gt.annToMask(ann) for ann in gt_anns]
        dt_masks = [coco_dt.annToMask(ann) for ann in dt_anns]
        gt_whistles = [mask_to_whistle(mask) for mask in gt_masks]
        dt_whistles = [mask_to_whistle(mask) for mask in dt_masks]
        if debug:
            assert len(gt_whistles) == len(dt_whistles), f"gt and dt should have the \
            same number of whistles, gt: {len(gt_whistles)}, dt: {len(dt_whistles)}"
        
        img_to_whistles[img['id']] = {
            'gts': gt_whistles,
            'dts': dt_whistles,
            'w': w,
            'img_id': img_id,
        }
    sum_gts = sum([len(whistles['gts']) for whistles in img_to_whistles.values()])
    rprint(f'gathered {len(img_to_whistles)} images with {sum_gts} whistles')
    return img_to_whistles


def compare_whistles(gts, dts, w, img_id, debug=False):
    """given whistle gt and dt in evaluation unit and get comparison results
    Args:
        gts, dts: N, 2 in format of y, x(or t, f)
    """
    gt_num = len(gts)
    gt_ranges = np.zeros((gt_num, 2))
    gt_durations = np.zeros(gt_num)

    dt_num = len(dts)
    dt_ranges = np.zeros((dt_num, 2))
    dt_durations = np.zeros(dt_num)
    
    if debug:
        for i in range(dt_num):
           assert (gts[i]== dts[i]).all(), "gt and dt should not be the same"

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
    gt_matched_all = []
    gt_missed_all = []
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
                        | (dt_start_xs>= gt_start_x) & (dt_end_xs <= gt_end_x)
        ovlp_dt_ids = np.nonzero(ovlp_cond)[0]

        matched = False
        deviations = []
        covered = 0
        # cnt = 0
        # dt_matched = []
        # dt_matched_dev = []
        # Note remove duration filter short < 75 pix whistle
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
            # if debug:
            #     deviation_tolerence = 0.1
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
                
        # if debug and cnt > 1:
        #     rprint(f"img_id: {img_id}, multiplication gt_id: {gt_idx} cnt: {cnt}")
        #     rprint(f"gt: {gt.T}",)
        #     for i, idx in enumerate(dt_matched):
        #         rprint(f"dt_idx: {idx}")
        #         rprint(f"dt: {dts[idx].T}")
        #         print(f"deviation: {dt_matched_dev[i].mean()}")
        
        if matched:
            gt_matched_all.append(gt_idx)
        else:
            gt_missed_all.append(gt_idx)

        if matched:
            gt_deviation = np.mean(deviations)
            all_deviation.append(gt_deviation) 
            all_covered.append(covered)
            all_dura.append(gt_dura)

    if debug:
        if dt_false_pos_all:
            for dt_fp_idx in dt_false_pos_all:
                rprint(f'img_id: {img_id}, dt_id:{dt_fp_idx}, fp_num: {len(dt_false_pos_all)}')
                rprint(f'dt: {dts[dt_fp_idx]}')
                rprint(f'gt: {gts[dt_fp_idx]}')
    res = {
        # TODO TP and FN are calculated based on dt and gt respectively
        'dt_false_pos_all': len(dt_false_pos_all),
        'dt_true_pos_all': len(dt_true_pos_all),
        'gt_matched_all': len(gt_matched_all),
        'gt_missed_all': len(gt_missed_all),
        'all_deviation': all_deviation,
        'all_covered': all_covered,
        'all_dura': all_dura
    }
    return res


def accumulate_wistle_results(img_to_whistles, debug=False):
    accumulated_res = {
        'dt_false_pos_all': 0,
        'dt_true_pos_all': 0,
        'gt_matched_all': 0,
        'gt_missed_all': 0,
        'all_deviation': [],
        'all_covered': [],
        'all_dura': []
    }
    for img_id, whistles in img_to_whistles.items():
        res = compare_whistles(**whistles, debug=debug)
        accumulated_res['dt_false_pos_all'] += res['dt_false_pos_all']
        accumulated_res['dt_true_pos_all'] += res['dt_true_pos_all']
        accumulated_res['gt_matched_all'] += res['gt_matched_all']
        accumulated_res['gt_missed_all'] += res['gt_missed_all']
        accumulated_res['all_deviation'].extend(res['all_deviation'])
        accumulated_res['all_covered'].extend(res['all_covered'])
        accumulated_res['all_dura'].extend(res['all_dura'])
    accumulated_res['all_deviation'] = np.mean(accumulated_res['all_deviation']).item()
    accumulated_res['all_covered'] = np.sum(accumulated_res['all_covered']).item()
    accumulated_res['all_dura'] = np.sum(accumulated_res['all_dura']).item()
    return accumulated_res

def sumerize_whisle_results(accumulated_res):
    """sumerize the whistle results"""
    dt_fp = accumulated_res['dt_false_pos_all']
    dt_tp = accumulated_res['dt_true_pos_all']
    gt_tp = accumulated_res['gt_matched_all']
    gt_fn = accumulated_res['gt_missed_all']

    precision = dt_tp / (dt_tp + dt_fp) if (dt_tp + dt_fp) > 0 else 0
    recall = gt_tp / (gt_tp + gt_fn) if (gt_tp + gt_fn) > 0 else 0
    frag = dt_tp / gt_tp if gt_tp > 0 else 0
    coverage = accumulated_res['all_covered'] / accumulated_res['all_dura'] if accumulated_res['all_dura'] > 0 else 0

    summary = {
        'precision': precision,
        'recall': recall,
        'frag': frag,
        'coverage': coverage
    }
    return summary


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Evaluate SAM on COCO dataset")
    parser.add_argument("--model_name", type=str, default="sam")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    known_args, unknown_args = parser.parse_known_args()

    cfg = tyro.cli(Config, args=unknown_args)

    # bbox_dts, mask_dts, gt_coco = get_detections(cfg, model_name=known_args.model_name, debug=known_args.debug)
    # with open(f'outputs/dt_result_{known_args.model_name}{'_debug' if known_args.debug else ''}.pkl', 'wb') as f:
    #     pickle.dump({'bbox_coco': bbox_dts, 'mask_coco': mask_dts, 'gt_coco': gt_coco}, f)

    with open(f'outputs/dt_result_{known_args.model_name}{'_debug' if known_args.debug else ''}.pkl', 'rb') as f:
        dt_results = pickle.load(f)
        gt_coco = dt_results['gt_coco']
        mask_dts = dt_results['mask_coco']

    img_to_whistles = gather_whistles(gt_coco, mask_dts, debug=known_args.debug)
    res = accumulate_wistle_results(img_to_whistles, debug=known_args.debug)
    rprint(res)
    summary = sumerize_whisle_results(res)
    rprint(summary)

    

