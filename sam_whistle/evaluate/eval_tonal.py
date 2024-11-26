import tyro
import json
import os
import numpy as np
from prettytable import PrettyTable
from itertools import chain
import cv2 

from sam_whistle.config import TonalConfig
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import TonalTracker
from sam_whistle.utils import utils


def eval_graph_search(cfg: TonalConfig, model=None, stems=None):
    print(f"{'#'*30} Evaluating {model} model {'#'*30}")
    if stems is None:
        stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))['test']
    elif isinstance(stems, str):
        stems = [stems]
    table = PrettyTable()
    table.field_names = ["Stem", "GT_N", "Precision", "Recall", "Deviation", "Coverage", "Frag"]
    all_res = {
        'dt_false_pos_all': [],
        'dt_true_pos_valid': [],
        'gt_matched_valid': [],
        'gt_missed_valid': [],
        'all_deviation': [],
        'all_covered_s': [],
        'all_dura': [],
    }
    for stem in stems:
        tracker = TonalTracker(cfg, stem)
        if model == 'sam':
            tracker.sam_inference()
        elif model == 'deep':
            tracker.dw_inference()
        elif model == 'fcn_spect':
            tracker.fcn_spect_inference()
        elif model == 'fcn_encoder':
            tracker.fcn_encoder_inference()
        elif model is None:
            pass
        else:
            raise ValueError("Invalid model name")

        tracker.build_graph()
        tonals = tracker.get_tonals()
        res = tracker.compare_tonals()
        # for k, v in res.items():
        #     print(f'{k}: {len(v)}')
        for key in all_res.keys():
            all_res[key].append(res[key])


        precision_valid = len(res['dt_true_pos_valid']) / (len(res['dt_true_pos_valid']) + len(res['dt_false_pos_all']))
        gt_num = (len(res['gt_matched_valid']) + len(res['gt_missed_valid']))
        recall_valid = len(res['gt_matched_valid']) / gt_num
        dev_mean = np.mean(res['all_deviation'])
        dev_std = np.std(res['all_deviation'])
        coverage = np.array(res['all_covered_s']) / np.array(res['all_dura'])
        coverage_mean = np.mean(coverage)
        coverage_std = np.std(coverage)
        frag = len(res['dt_true_pos_valid']) / len(res['gt_matched_valid'])
        table.add_row([stem, gt_num, f'{precision_valid*100:.2f}', f'{recall_valid*100:.2f}', f'{dev_mean:.2f}±{dev_std:.2f}', f'{coverage_mean:.2f}±{coverage_std:.2f}', f'{frag:.2f}'])

        if cfg.visualize:
            shape = tracker.spect_raw.shape
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
                
                utils.visualize_array(raw_block,  cmap='gray', filename=f'{col}_raw', save_dir=f'outputs/eval2/{stem}')
                utils.visualize_array(raw_block, cmap='gray', mask=(pred_block > tracker.thre).astype(int), random_colors=False, mask_alpha=1, filename=f'{col}_pred', save_dir=f'outputs/eval2/{stem}')

                raw_block = cv2.cvtColor(raw_block, cv2.COLOR_GRAY2RGB)
                non_zero_mask = np.any(gt_tonal_block != 0, axis=-1)
                overlay_image = raw_block.copy()
                alpha = 0.8
                overlay_image[non_zero_mask] = (1 - alpha) * raw_block[non_zero_mask] + alpha * gt_tonal_block[non_zero_mask]
                utils.visualize_array(overlay_image, cmap='gray', filename=f'{col}_gt_tonal', save_dir=f'outputs/eval2/{stem}')

                non_zero_mask = np.any(pred_tonal_block != 0, axis=-1)
                overlay_image = raw_block.copy()
                overlay_image[non_zero_mask] = (1 - alpha) * raw_block[non_zero_mask] + alpha * pred_tonal_block[non_zero_mask]

                utils.visualize_array(overlay_image, cmap='gray', filename=f'{col}_pred_tonal', save_dir=f'outputs/eval2/{stem}')



    all_dt_false_pos_all = list(chain.from_iterable(all_res['dt_false_pos_all']))
    all_dt_true_pos_valid = list(chain.from_iterable(all_res['dt_true_pos_valid']))
    all_gt_matched_valid = list(chain.from_iterable(all_res['gt_matched_valid']))
    all_gt_missed_valid = list(chain.from_iterable(all_res['gt_missed_valid']))
    all_deviation = np.array(list(chain.from_iterable(all_res['all_deviation'])))
    all_covered_s = np.array(list(chain.from_iterable(all_res['all_covered_s'])))
    all_dura = np.array(list(chain.from_iterable(all_res['all_dura'])))

    all_precision = len(all_dt_true_pos_valid) / (len(all_dt_true_pos_valid)+len(all_dt_false_pos_all))
    all_recall = len(all_gt_matched_valid) / (len(all_gt_matched_valid) + len(all_gt_missed_valid))
    all_dev_mean = np.mean(all_deviation)
    all_dev_std = np.std(all_deviation)
    all_coverage = all_covered_s / all_dura
    all_coverage_mean = np.mean(all_coverage)
    all_coverage_std = np.std(all_coverage)
    all_frag = len(all_dt_true_pos_valid) / len(all_gt_matched_valid)
    all_gt_num = len(all_gt_matched_valid) + len(all_gt_missed_valid)
    table.add_row(["Overall", all_gt_num, f'{all_precision*100:.2f}', f'{all_recall*100:.2f}', f'{all_dev_mean:.2f}±{all_dev_std:.2f}', f'{all_coverage_mean:.2f}±{all_coverage_std:.2f}', f'{all_frag:.2f}'])
    print(table)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str)
    args, remainings = parser.parse_known_args()
    cfg = tyro.cli(TonalConfig, args=remainings)
    
    stem = "Qx-Dc-CC0411-TAT11-CH2-041114-154040-s"
    eval_graph_search(cfg, model = args.model, stems=stem)