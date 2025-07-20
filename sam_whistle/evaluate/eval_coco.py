from collections import defaultdict
from copy import deepcopy
import glob
import json
import os
import time
from typing import List
import cv2
import librosa
from scipy.signal import medfilt2d
from tqdm import tqdm
import tyro
import argparse
import pickle
from rich import print as rprint
import copy
from dataclasses import dataclass

import numpy as np
import pycocotools.mask as maskUtils
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from shapely.geometry import LineString, Point
from shapely.ops import clip_by_rect
from skimage.morphology import skeletonize
import yaml


from sam_whistle.datasets.whistle_coco import WhistleCOCO
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import *
from sam_whistle.evaluate.tonal_extraction.write_binary import writeTimeFrequencyBinary, writeContoursBinary


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

def pix_to_tf(pix, height = 769):
    """Convert pixel coordinates to time-frequency coordinates."""
    time = (pix[:, 0]-0.5) * 0.002  # Convert to seconds
    freq = (height - 1 - pix[:, 1] + 0.5) * 125  # Convert to frequency in Hz
    return np.column_stack((time, freq))


@dataclass
class contour:
    time: float
    freq: float

def tonal_save(stem, tonals, tonals_snr=None, model_name = 'mask2former'):
    """Save the tonnals to a silbido binary file
    Args:
        tonnals: list of tonnals array
        preprcoess to list of dictionaries of tonnals in format
        {"tfnodes": [
                {"time": 3.25, "freq": 50.125, "snr": 6.6, "phase": 0.25, "ridge": 1.0},
                {"time":...},
                ...,]
        }
    """
    # convert to dataclass
    filename = f'outputs/{stem}_{model_name}_dt.bin'
    if tonals_snr is None:
        tonals_ = [contour(time=tonal[:, 0], freq=tonal[:, 1]) for tonal in tonals]
        writeTimeFrequencyBinary(filename, tonals_)
    else:
        tonals_ = [{'tfnodes': [{'time': tf[0], 'freq': tf[1], 'snr': snr} for tf, snr in zip(tfs, snrs)]} for tfs, snrs in zip(tonals, tonals_snr)]
        writeContoursBinary(filename, tonals_, snr=True)


def get_detections_record(cfg, model_name, output_bin= False, debug=False):
    """Get the detection of each record"""
    tic = time.time()
    stems = yaml.safe_load(open(os.path.join(cfg.root_dir, cfg.meta_file)))
    stems = stems['test'] #+ stems['train']  # test imgs spread over origin train and test audio
    
    # stems = stems[:1]
    # stems = ['palmyra092007FS192-070924-205305']

    if debug:
        stems = ['Qx-Tt-SCI0608-N1-060814-121518']
        pass

    # trackers = {}
    img_to_whistles = dict()
    scores_all = []

    rprint(f'Starting to get detections from {stems}')
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
        if output_bin:
            tonal_save(stem, dt_tonals, model_name=model_name)

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
            _, bound = get_traj_valid((tracker.raw_spect*255).numpy().astype(np.uint8), traj_pix)
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
    
    tok = time.time() - tic
    rprint(f'Finished gathering detections in {tok:.2f} seconds')
    return img_to_whistles


def get_detections_coco(cfg, model_name='sam', debug=False):
    whistle_coco_data = os.path.join(cfg.root_dir, 'coco/test/data')
    whistle_coco_label = os.path.join(cfg.root_dir, 'coco/test/labels.json')

    test_set = WhistleCOCO(root=whistle_coco_data, annFile=whistle_coco_label)
    gt_coco = test_set.coco

    stems = yaml.safe_load(open(os.path.join(cfg.root_dir, cfg.meta_file)))
    stems = stems['test'] #+ stems['train']  # test imgs spread over origin train and test audio
    
    # stems = stems[:1]
    # stems = ['palmyra092007FS192-070924-205305']

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
    rprint(f'Starting to get detections from {stems}')
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

        rprint(f"stem: {stem}, gt_image_ids: {len(stem_img_ids)}")
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
            # rprint(f"stem: {stem}, dt_trajs: {len(dt_trajs)}")
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

    return bbox_coco, mask_coco, gt_coco, gt_image_ids


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


def gather_whistles(coco_gt:COCO, coco_dt:COCO, gt_image_ids, filter_dt=0, valid_gt=False, root_dir=None, debug=False, model_name='sam'):
    """gather per image whistles from instance masks"""
    
    if debug:
        coco_dt = deepcopy(coco_gt)
        for ann in coco_dt.anns.values():
            ann['score'] = 1.0

    img_to_whistles = dict()
    for img_id in tqdm(gt_image_ids, desc='gathering whistles'):
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
            if model_name == 'sam':
                if score > filter_dt:
                    dt_masks.append(mask)
                else:
                    continue
            elif model_name == 'dw':
                dt_masks.append(mask)

        gt_whistles = [mask_to_whistle(mask) for mask in gt_masks]
        
        # bounds = []
        # gt_whistles_ = []
        # img_info = coco_gt.imgs[img_id]
        # image = cv2.imread(os.path.join(root_dir,'coco/test/data' , img_info['file_name']))
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


def wave_to_spect(
        waveform, 
        sample_rate=None,
        frame_ms=8, 
        hop_ms=2, 
        pad=0, 
        n_fft=None, 
        hop_length=None, 
        top_db=None, 
        center = False, 
        amin = 1e-16, 
        **kwargs
    ):
    """Convert waveform to raw spectrogram in power dB scale."""
    # fft params
    if n_fft is None:
        if frame_ms is not None and sample_rate is not None:
            n_fft = int(frame_ms * sample_rate / 1000)
        else:
            raise ValueError("n_fft or frame_ms must be provided.")
    if hop_length is None:
        if hop_ms is not None and sample_rate is not None:
            hop_length = int(hop_ms * sample_rate / 1000)
        else:
            raise ValueError("hop_length or hop_ms must be provided.")

    # spectrogram magnitude
    spect = librosa.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hamming',
        center=center,
        pad_mode='reflect',
    )
    # decibel scale spectrogram with cutoff specified by top_db
    spect_power_db = librosa.amplitude_to_db(
        np.abs(spect),
        ref=1.0,
        amin=amin,
        top_db=top_db,
    )
    return spect_power_db # (freq, time)

def load_wave_file(file_path, type='numpy'):
    """Load one wave file."""
    if type == 'numpy':
        waveform, sample_rate = librosa.load(file_path, sr=None)
    elif type == 'tensor':
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def snr_spect(spect_db, click_thr_db, broadband_thr_n):
    meanf_db = np.mean(spect_db, axis=1, keepdims=True)
    click_p = np.sum((spect_db - meanf_db) > click_thr_db, axis=0) > broadband_thr_n
    use_p = ~click_p
    spect_db = medfilt2d(spect_db, kernel_size=[3,3])
    if np.sum(use_p) == 0:
        # Qx-Dc-SC03-TAT09-060516-173000.wav 4500 no use_p
        use_p = np.ones_like(click_p)
    meanf_db = np.mean(spect_db[:, use_p], axis=1, keepdims=True)
    snr_spect_db = spect_db - meanf_db
    return snr_spect_db

def compare_whistles(gts, dts, w, img_id, boudns_gt=None, valid_gt = False, valid_len = 75, deviation_tolerence = 350/125, debug=False):
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
    
    if type(valid_len) == int:
        delt= 1
    else:
        delt = 0.002

    for gt_idx, gt in enumerate(gts):
        gt_start_x = max(0, gt[:, 0].min())
        gt_end_x = min(w -delt , gt[:, 0].max())
        gt_dura = gt_end_x + delt - gt_start_x  # add 1 in pixel
        gt_durations[gt_idx] = gt_dura
        gt_ranges[gt_idx] = (gt_start_x, gt_end_x)
  
    for dt_idx, dt in enumerate(dts):
        dt_start_x = max(0, dt[:, 0].min())
        dt_end_x = min(w, dt[:, 0].max())
        dt_ranges[dt_idx] = (dt_start_x, dt_end_x)
        dt_durations[dt_idx] = dt_end_x + delt - dt_start_x # add 1 in pixel
    
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

    if valid_gt or debug:
        waveform, sample_rate = load_wave_file(f'data/cross/audio/{img_id}.wav')
        spect_power_db= wave_to_spect(waveform, sample_rate)
        H, W = spect_power_db.shape[-2:]
        spect_snr = np.zeros_like(spect_power_db)
        block_size = 1500  # 300ms
        broadband = 0.01
        search_row = 4
        ratio_above_snr = 0.3
        for i in range(0, H, block_size):
            spect_snr[i:i+block_size] = snr_spect(spect_power_db[i:i+block_size], click_thr_db=10, broadband_thr_n=broadband*H )
        spect_snr = np.flipud(spect_snr) # flip frequency axis, low freq at the bottom

    # go through each ground truth
    for gt_idx, gt in enumerate(gts):
        gt_start_x, gt_end_x = gt_ranges[gt_idx]
        gt_dura = gt_durations[gt_idx]
        # Note: remove interpolation and snr validation that filter low gt snr level
        # which requires addition input
        dt_start_xs, dt_end_xs = dt_ranges[:, 0], dt_ranges[:, 1]
        ovlp_cond = (dt_start_xs <= gt_start_x) & (dt_end_xs >= gt_start_x) \
                        | (dt_start_xs>= gt_start_x) & (dt_start_xs <= gt_end_x)
        ovlp_dt_ids = np.nonzero(ovlp_cond)[0]

        matched = False
        deviations = []
        covered = 0
        # Note remove duration filter short < 75 pix whistle
        valid = True
        if valid_gt:
            search_row_low = np.minimum(np.maximum(gt[:, 1] - search_row, 0), H)
            search_row_high = np.maximum(np.minimum(gt[:, 1] + search_row, H), 0)
            gt_cols = gt[:, 0][gt[:, 0] < W]
            try:
                spec_search = [np.max(spect_snr[l:h, col]).item() for i, (l,h, col) in enumerate(zip(search_row_low, search_row_high, gt_cols))] 
            except: 
                import pdb; pdb.set_trace()
            sorted_search_snr = np.sort(spec_search)
            bound_idx = max(0, round(len(sorted_search_snr) * (1- ratio_above_snr))-1)
            gt_snr = sorted_search_snr[bound_idx]
            if gt_dura < valid_len or gt_snr < 3:
                valid= False


        for ovlp_dt_idx in ovlp_dt_ids:
            ovlp_dt = dts[ovlp_dt_idx]
            dt_xs, dt_ys = ovlp_dt[:, 0], ovlp_dt[:, 1]
            dt_ovlp_x_idx = np.nonzero((dt_xs >= gt_start_x) & (dt_xs <= gt_end_x))[0]
            dt_ovlp_xs = dt_xs[dt_ovlp_x_idx]
            dt_ovlp_ys = dt_ys[dt_ovlp_x_idx]

            # Note: remove interpolation
            gt_ovlp_ys = gt[:, 1][np.searchsorted(gt[:, 0], dt_ovlp_xs)]
            deviation = np.abs(gt_ovlp_ys - dt_ovlp_ys)
            deviation_tolerence = deviation_tolerence
            if len(deviation)> 0 and np.mean(deviation) <= deviation_tolerence:
                matched = True
                
                if ovlp_dt_idx in dt_false_pos_all:
                    dt_false_pos_all.remove(ovlp_dt_idx)
                # TODO: has multiplications
                dt_true_pos_all.append(ovlp_dt_idx)  
                # TODO: the deviation and coverage of same overlap can be counted multiple times
                deviations.extend(deviation)

                # dt_matched_dev.append(deviation)
                
                covered += dt_ovlp_xs.max() - dt_ovlp_xs.min() + delt
                if valid:
                    dt_true_pos_valid.append(ovlp_dt_idx)
        
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
        all_dura.append(gt_dura) # move out from matched

    if debug:
        freq_height = 769
        dt_false_pos_tf_all = [pix_to_tf(dts[idx], height=freq_height) for idx in dt_false_pos_all]
        tonals_snr = [spect_snr[dts[idx][:, 1].astype(int),dts[idx][:, 0].astype(int)] for idx in dt_false_pos_all]
        tonal_save(img_id, dt_false_pos_tf_all, tonals_snr, 'mask2former_swin_fp')
        dt_snrs = [np.mean(snr) for snr in tonals_snr]

        dt_false_neg_tf = [pix_to_tf(gts[idx], height=freq_height) for idx in gt_missed_all]
        tonal_save(img_id, dt_false_neg_tf, model_name='mask2former_swin_fn')

        if len(dt_snrs) > 0:
            # rprint({i+1: dt_snrs[i].item() for i in range(len(dt_snrs))})
            rprint(f'stem: {img_id}, min_snr: {np.min(dt_snrs)}, max_snr: {np.max(dt_snrs)}, mean:{np.mean(dt_snrs)}, above 9: {np.sum(np.array(dt_snrs) > 9)}')
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


def accumulate_wistle_results(img_to_whistles, valid_gt, valid_len=75,deviation_tolerence = 350/125, debug=False):
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
        res = compare_whistles(**whistles, valid_gt = valid_gt, valid_len = valid_len, deviation_tolerence = deviation_tolerence, debug=debug)
        rprint(f'img_id: {img_id}')
        rprint(summarize_whistle_results(res))
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

def summarize_whistle_results(accumulated_res):
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
        'gt_all': gt_tp + gt_fn,
        'dt_all': dt_tp + dt_fp,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
        'frag': frag,
        'coverage': coverage,
        'gt_n':(gt_tp_valid + gt_fn_valid),
        'dt_n':(dt_tp_valid + dt_fp),
        'precision_valid': precision_valid,
        'recall_valid': recall_valid,
        'frag_valid': frag_valid,
        'f1_valid': 2 * precision_valid * recall_valid / (precision_valid + recall_valid) if (precision_valid + recall_valid) > 0 else 0
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SAM on COCO dataset")
    parser.add_argument("--model_name", type=str, default="sam")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--type", type=str, default="coco")
    parser.add_argument("--output_bin", action="store_true", help="Output bin file")
    known_args, unknown_args = parser.parse_known_args()

    cfg = tyro.cli(Config, args=unknown_args)
    # known_args.debug = True
    # cfg.thre_norm = 0.06939393939393938
    # cfg.thre_norm = 0.05
    if known_args.type == 'coco':
        bbox_dts, mask_dts, gt_coco, gt_image_ids = get_detections_coco(cfg, model_name=known_args.model_name, debug=known_args.debug)
        
        data_name = cfg.root_dir.split('/')[-1]

        with open(f'outputs/dt_result_{data_name}_{known_args.model_name}{'_debug' if known_args.debug else ''}.pkl', 'wb') as f:
            pickle.dump({'bbox_coco': bbox_dts, 'mask_coco': mask_dts, 'gt_coco': gt_coco}, f)

        with open(f'outputs/dt_result_{data_name}_{known_args.model_name}{'_debug' if known_args.debug else ''}.pkl', 'rb') as f:
            dt_results = pickle.load(f)
            gt_coco = dt_results['gt_coco']
            mask_dts = dt_results['mask_coco']

        img_to_whistles = gather_whistles(gt_coco, mask_dts, gt_image_ids, root_dir=cfg.root_dir, debug=known_args.debug, model_name=known_args.model_name)
    elif known_args.type == 'record':
        img_to_whistles = get_detections_record(cfg, model_name=known_args.model_name, output_bin= known_args.output_bin, debug=known_args.debug)
    elif known_args.type == 'coco_record':
        pass
    else:
        raise ValueError(f"Unknown type: {known_args.type}")
 

    res = accumulate_wistle_results(img_to_whistles, valid_gt=True, debug=known_args.debug)
    summary = summarize_whistle_results(res)
    rprint(f'evaluation based on unit {known_args.type}')
    rprint(summary)

    

