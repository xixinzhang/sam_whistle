from collections import defaultdict
import pickle
import tyro
import json
import os
import numpy as np
from prettytable import PrettyTable
from itertools import chain
import cv2 
from dataclasses import dataclass, asdict

from sam_whistle.config import TonalConfig
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import TonalTracker, TonalResults
from sam_whistle.utils import utils


@dataclass
class TonalStats:
    precision_valid: float
    recall_valid: float
    dev_mean: float
    dev_std: float
    coverage_mean: float
    coverage_std: float
    frag: float
    gt_num: int


def res_to_stats(res: TonalResults):
    precision_valid = res.dt_true_pos_valid / (res.dt_true_pos_valid + res.dt_false_pos_all)
    gt_num = res.gt_matched_valid + res.gt_missed_valid
    recall_valid = res.gt_matched_valid / gt_num
    dev_mean = np.mean(res.all_deviation)
    dev_std = np.std(res.all_deviation)
    coverage = np.array(res.all_covered_s) / np.array(res.all_dura)
    coverage_mean = np.mean(coverage)
    coverage_std = np.std(coverage)
    frag = res.dt_true_pos_valid / res.gt_matched_valid

    stats = TonalStats(
        precision_valid=precision_valid,
        recall_valid=recall_valid,
        dev_mean=dev_mean,
        dev_std=dev_std,
        coverage_mean=coverage_mean,
        coverage_std=coverage_std,
        frag=frag,
        gt_num=gt_num
    )
    return stats

def extract_tonals(tracker: TonalTracker, thre, visualize=False, output_dir=None, stem=None):
    tracker.reset()
    tracker.thre = thre
    tracker.build_graph()
    tonals = tracker.get_tonals()
    res = tracker.compare_tonals()


    if visualize:
        gt_tonals =[]
        for anno in tracker.gt_tonals:
            gt = utils.anno_to_spect_point(anno)
            gt_tonals.append(gt)
        colored_map_gt = utils.get_colored_tonal_map(tracker.origin_shape, gt_tonals)
        kernel = np.ones((3, 3), np.uint8)
        colored_map_gt = cv2.dilate(colored_map_gt, kernel, iterations=1)
        colored_map_gt = colored_map_gt[-tracker.cfg.spect_cfg.crop_top: -tracker.cfg.spect_cfg.crop_bottom+1]
        
        pred_tonals = []
        for anno in tonals:
            pred = utils.anno_to_spect_point(anno)
            pred_tonals.append(pred)
        colored_map_pred = utils.get_colored_tonal_map(tracker.origin_shape, pred_tonals)
        colored_map_pred = cv2.dilate(colored_map_pred, kernel, iterations=1)
        colored_map_pred = colored_map_pred[-tracker.cfg.spect_cfg.crop_top: -tracker.cfg.spect_cfg.crop_bottom+1]

        for col in tracker.start_cols:
            raw_block = tracker.spect_raw[:, col: col + tracker.block_size]
            pred_block = tracker.spect_map[::-1, col: col + tracker.block_size]
            gt_tonal_block = colored_map_gt[:, col: col + tracker.block_size]
            pred_tonal_block = colored_map_pred[:, col: col + tracker.block_size]
            
            utils.visualize_array(raw_block,  cmap='gray', filename=f'{col}_raw', save_dir=f'{output_dir}/{stem}')
            utils.visualize_array(raw_block, cmap='gray', mask=(pred_block > tracker.thre).astype(int), random_colors=False, mask_alpha=1, filename=f'{col}_pred', save_dir=f'{output_dir}/{stem}')

            raw_block = cv2.cvtColor(raw_block, cv2.COLOR_GRAY2RGB)
            non_zero_mask = np.any(gt_tonal_block != 0, axis=-1)
            overlay_image = raw_block.copy()
            alpha = 0.8
            overlay_image[non_zero_mask] = (1 - alpha) * raw_block[non_zero_mask] + alpha * gt_tonal_block[non_zero_mask]
            utils.visualize_array(overlay_image, cmap='gray', filename=f'{col}_gt_tonal', save_dir=f'{output_dir}/{stem}')

            non_zero_mask = np.any(pred_tonal_block != 0, axis=-1)
            overlay_image = raw_block.copy()
            overlay_image[non_zero_mask] = (1 - alpha) * raw_block[non_zero_mask] + alpha * pred_tonal_block[non_zero_mask]

            utils.visualize_array(overlay_image, cmap='gray', filename=f'{col}_pred_tonal', save_dir=f'{output_dir}/{stem}')

    return res


def eval_graph_search(cfg: TonalConfig, model_name='graph_search', stems=None, output_dir='output/', min_thre=0.01, max_thre=0.99, thre_num=10):
    print(f"{'#'*30} Evaluating {model_name} model {'#'*30}")
    if stems is None:
        stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))['test']
    elif isinstance(stems, str):
        stems = [stems]


    threshold_list = np.linspace(min_thre, max_thre, thre_num, endpoint=True)
    pr = defaultdict(list)
    all_tonal_stats = defaultdict(dict[str, TonalStats])
    all_tonal_res = defaultdict(dict)

    for stem in stems:
        tracker = TonalTracker(cfg, stem)
        if model_name == 'sam':
            tracker.sam_inference()
        elif model_name == 'deep':
            tracker.dw_inference()
        elif model_name == 'fcn_spect':
            tracker.fcn_spect_inference()
        elif model_name == 'fcn_encoder':
            tracker.fcn_encoder_inference()
        elif model_name == 'graph_search':
            assert max_thre > 1, "Threshold should be largeer than 1 for orginal spectrogram"
        else:
            raise ValueError("Invalid model name")
        print(f'{'#'*30} Finished Inference {'#'*30}')

        for thre in threshold_list:
            print(f"{'#'*30} Threshold: {thre} {'#'*30}")
            res = extract_tonals(tracker, thre, visualize=cfg.visualize, output_dir=output_dir+f'{thre}', stem=stem)
            all_tonal_res[thre][stem] = res
            stats = res_to_stats(res)
            all_tonal_stats[thre][stem] = stats
            precision = stats.precision_valid
            recall = stats.recall_valid
            pr[stem].append((precision, recall, thre))
            print(f'[{stem}] precision: {precision:.2f}, recall: {recall:.2f}, thre: {thre:.2f}')

    for thre, res_dict in all_tonal_res.items():
        res_all = TonalResults()
        for stem, res in res_dict.items():
            res_all.merge(res)
        stats = res_to_stats(res_all)
        all_tonal_stats[thre]['all'] = stats
        precision = stats.precision_valid
        recall = stats.recall_valid
        pr['all'].append((precision, recall, thre))
        # print(f'[All]: precision: {precision:.2f}, recall: {recall:.2f}, thre: {thre:.2f}')

    for thre in threshold_list:
        table = PrettyTable()
        table.field_names = ["Stem", "GT_N", "Precision", "Recall", "Deviation", "Coverage", "Frag"]
        for stem, stats in all_tonal_stats[thre].items():
            table.add_row([stem, stats.gt_num, f'{stats.precision_valid*100:.2f}', f'{stats.recall_valid*100:.2f}', f'{stats.dev_mean:.2f}±{stats.dev_std:.2f}', f'{stats.coverage_mean:.2f}±{stats.coverage_std:.2f}', f'{stats.frag:.2f}'])
        print(f'threreshold: {thre}')
        print(table)

    eval_res = utils.eval_tonal_map(pr, model_name)
    with open(os.path.join(cfg.log_dir, f'{model_name}_results_tonal.pkl'), 'wb') as f:
        pickle.dump(eval_res, f)
    utils.plot_pr_curve([eval_res], cfg.log_dir, f'pr_curve_tonal.png')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_multiple', action = 'store_true', help='Evaluate a single model')
    parser.add_argument('--model', default='graph_search', type=str)
    parser.add_argument('--min_thre', type=float, default=0.01, help='Minimum threshold for filtering')
    parser.add_argument('--max_thre', type=float, default=0.99, help='Maximum threshold for filtering')
    parser.add_argument('--thre_num', type=int, default=20, help='Number of thresholds') 
    args, remainings = parser.parse_known_args()
    cfg = tyro.cli(TonalConfig, args=remainings)
    
    if not args.eval_multiple:
        stem = None
        # stem = "Qx-Dc-CC0411-TAT11-CH2-041114-154040-s"
        eval_graph_search(cfg, model_name = args.model, stems=stem, min_thre=args.min_thre, max_thre=args.max_thre, thre_num=args.thre_num)
    else:
        eval_results = [
            'logs/11-23-2024_15-19-19-sam/sam_results_tonal.pkl',
            'logs/11-23-2024_15-27-33-deep_whistle/deep_results_tonal.pkl',
            # 'logs/11-23-2024_15-39-59-fcn_spect/fcn_spect_results_tonal.pkl',
            # 'logs/11-24-2024_03-02-50-fcn_encoder_imbalance/fcn_encoder_results_tonal.pkl'
            'logs/graph_search/graph_search_results_tonal.pkl'
        ]
        eval_res_li = [pickle.load(open(res_file, 'rb')) for res_file in eval_results]
        utils.plot_pr_curve(eval_res_li, 'logs', 'pr_curve_tonal.png')