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
    excess_mean: float
    excess_std: float
    frag: float
    gt_num: int


def res_to_metric(res: TonalResults):
    precision_valid = res.dt_true_pos_valid / (res.dt_true_pos_valid + res.dt_false_pos_all)
    gt_num = res.gt_matched_valid + res.gt_missed_valid
    recall_valid = res.gt_matched_valid / gt_num
    dev_mean = np.mean(res.all_deviation)
    dev_std = np.std(res.all_deviation)
    coverage = np.array(res.all_covered_s) / np.array(res.all_dura)
    coverage_mean = np.mean(coverage)
    coverage_std = np.std(coverage)
    excess = np.array(res.all_excess_s) / np.array(res.all_dura)
    excess_mean = np.mean(excess)
    excess_std = np.std(excess)
    frag = res.dt_true_pos_valid / res.gt_matched_valid

    stats = TonalStats(
        precision_valid=precision_valid,
        recall_valid=recall_valid,
        dev_mean=dev_mean,
        dev_std=dev_std,
        coverage_mean=coverage_mean,
        coverage_std=coverage_std,
        excess_mean=excess_mean,
        excess_std=excess_std,
        frag=frag,
        gt_num=gt_num
    )
    return stats

def extract_tonals(tracker: TonalTracker, thre, visualize=False, output_dir=None, stem=None):
    tracker.reset()
    tracker.thre = thre
    peaks = tracker.build_graph()
    tonals = tracker.get_tonals()
    res = tracker.compare_tonals()

    if visualize:
        gt_tonals =[]
        for anno in tracker.gt_tonals_missed_valid:
            gt = utils.anno_to_spect_point(anno)
            gt_tonals.append(gt)
        gt_tonal_mask = utils.get_tonal_mask(tracker.origin_shape, gt_tonals)
        kernel = np.ones((3, 3), np.uint8)
        gt_tonal_mask = cv2.dilate(gt_tonal_mask, kernel, iterations=1).astype(int)
        gt_tonal_mask = gt_tonal_mask[-tracker.cfg.spect_cfg.crop_top: -tracker.cfg.spect_cfg.crop_bottom+1]
        
        pred_tonals = []
        for anno in tonals:
            pred = utils.anno_to_spect_point(anno)
            pred_tonals.append(pred)
        pred_tonal_mask = utils.get_tonal_mask(tracker.origin_shape, pred_tonals)
        pred_tonal_mask = cv2.dilate(pred_tonal_mask, kernel, iterations=1).astype(int)
        pred_tonal_mask = pred_tonal_mask[-tracker.cfg.spect_cfg.crop_top: -tracker.cfg.spect_cfg.crop_bottom+1]
        for col in tracker.start_cols:
            raw_block = tracker.spect_raw[:, col: col + tracker.block_size]
            pred_block = tracker.spect_map[::-1, col: col + tracker.block_size]
            pred_mask = (pred_block > tracker.thre).astype(int)
            gt_tonal_block = gt_tonal_mask[:, col: col + tracker.block_size]
            pred_tonal_block = pred_tonal_mask[:, col: col + tracker.block_size]
            block_peaks = [(peak[0], peak[1]- col) for peak in peaks  if peak[1] >= col and peak[1] < col + tracker.block_size]
            
            utils.plot_spect(raw_block, filename=f'{col}_1.raw', save_dir=f'{output_dir}/{stem}')
            utils.plot_mask_over_spect(raw_block, pred_mask, filename=f'{col}_4.pred_conf', save_dir=f'{output_dir}/{stem}')
            utils.plot_mask_over_spect(raw_block, gt_tonal_block, filename=f'{col}_3.gt_tonal', save_dir=f'{output_dir}/{stem}', random_colors=True)
            utils.plot_mask_over_spect(raw_block, pred_tonal_block, filename=f'{col}_2.pred_tonal', save_dir=f'{output_dir}/{stem}', random_colors=True)
            if len(block_peaks) > 0:
                utils.plot_points_over_spect(raw_block, [block_peaks], filename=f'{col}_5.peaks', save_dir=f'{output_dir}/{stem}')
            else:
                utils.plot_spect(raw_block, filename=f'{col}_5.peaks', save_dir=f'{output_dir}/{stem}')
        
    return res


def eval_graph_search(cfg: TonalConfig, model_name='graph_search', stems=None, output_dir='outputs', min_thre=0.01, max_thre=0.99, thre_num=10):
    print(f"{'#'*30} Evaluating {model_name} model {'#'*30}")
    if stems is None:
        stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))['test']
    elif isinstance(stems, str):
        stems = [stems]


    threshold_list = np.linspace(min_thre, max_thre, thre_num, endpoint=True)
    threshold_list = [0.01, 0.47, 0.99]
    prs = defaultdict(list)
    # all_tonal_stats = defaultdict(dict[str, TonalStats])
    # all_tonal_res = defaultdict(dict)

    trackers = {}
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
        trackers[stem]= tracker
        print(f'{'#'*30} Finished Inference {'#'*30}')

    for thre in threshold_list:
        print(f"{'#'*30} Threshold: {thre} {'#'*30}")
        table = PrettyTable()
        table.field_names = ["Stem", "GT_N", "Precision", "Recall", "Deviation", "Coverage", "Excess", "Frag"]
        res_all = TonalResults()
        for stem, tracker in trackers.items():
            res = extract_tonals(tracker, thre, visualize=cfg.visualize, output_dir=output_dir+f'/{thre}', stem=stem)
            res_all.merge(res)
            metrics = res_to_metric(res)
            precision = metrics.precision_valid
            recall = metrics.recall_valid
            prs[stem].append((precision, recall, thre))
            print(f'[{stem}] precision: {precision:.2f}, recall: {recall:.2f}, thre: {thre:.2f}, ov_valid: {res.dt_true_pos_valid}, false_dt: {res.dt_false_pos_all}, gt_matched: {res.gt_matched_valid}, gt_missed: {res.gt_missed_valid}')
            table.add_row([stem, metrics.gt_num, f'{metrics.precision_valid*100:.2f}', f'{metrics.recall_valid*100:.2f}', f'{metrics.dev_mean:.2f}±{metrics.dev_std:.2f}', f'{metrics.coverage_mean:.2f}±{metrics.coverage_std:.2f}', f'{metrics.excess_mean:.2f}±{metrics.excess_std:.2f}' ,f'{metrics.frag:.2f}'])

        metrics_all = res_to_metric(res_all)
        prs['all'].append((metrics_all.precision_valid, metrics_all.recall_valid, thre))
        table.add_row(['All', metrics_all.gt_num, f'{metrics_all.precision_valid*100:.2f}', f'{metrics_all.recall_valid*100:.2f}', f'{metrics_all.dev_mean:.2f}±{metrics_all.dev_std:.2f}', f'{metrics_all.coverage_mean:.2f}±{metrics_all.coverage_std:.2f}',  f'{metrics_all.excess_std:.2f}±{metrics_all.excess_std:.2f}',f'{metrics_all.frag:.2f}'])
        print(f'threreshold: {thre}')
        print(table)

    for stem, pr in prs.items():
        eval_res = utils.eval_tonal_map(pr, model_name)
        with open(os.path.join(cfg.log_dir, f'{model_name}_results_tonal_{stem}.pkl'), 'wb') as f:
            pickle.dump(eval_res, f)
        utils.plot_pr_curve([eval_res], cfg.log_dir, f'pr_curve_tonal_{stem}.png')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_multiple', action = 'store_true', help='Evaluate a single model')
    parser.add_argument('--model', default='graph_search', type=str)
    parser.add_argument('--min_thre', type=float, default=0.01, help='Minimum threshold for filtering')
    parser.add_argument('--max_thre', type=float, default=0.99, help='Maximum threshold for filtering')
    parser.add_argument('--thre_num', type=int, default=20, help='Number of thresholds')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args, remainings = parser.parse_known_args()
    cfg = tyro.cli(TonalConfig, args=remainings)
    
    if not args.eval_multiple:
        stem = None
        # stem = ["Qx-Dc-CC0411-TAT11-CH2-041114-154040-s", "palmyra092007FS192-070924-205305"]
        # stem = "Qx-Dc-CC0411-TAT11-CH2-041114-154040-s"
        eval_graph_search(cfg, model_name = args.model, stems=stem, min_thre=args.min_thre, max_thre=args.max_thre, thre_num=args.thre_num, output_dir=args.output_dir)
    else:
        eval_results = [
            'logs/11-23-2024_15-19-19-sam/sam_results_tonal.pkl',
            'logs/11-23-2024_15-27-33-deep_whistle/deep_results_tonal.pkl',
            'logs/11-23-2024_15-39-59-fcn_spect/fcn_spect_results_tonal.pkl',
            # 'logs/11-24-2024_03-02-50-fcn_encoder_imbalance/fcn_encoder_results_tonal.pkl'
            'logs/graph_search/graph_search_results_tonal.pkl'
        ]
        eval_res_li = [pickle.load(open(res_file, 'rb')) for res_file in eval_results]
        utils.plot_pr_curve(eval_res_li, 'logs', 'pr_curve_tonal.png')