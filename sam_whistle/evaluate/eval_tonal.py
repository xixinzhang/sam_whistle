import tyro
import json
import os
import numpy as np
from prettytable import PrettyTable
from itertools import chain

from sam_whistle.config import TonalConfig
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import TonalTracker


def eval_graph_search(cfg: TonalConfig, model='sam'):
    stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))['test']
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
        tracker.build_graph()
        tonals = tracker.get_tonals()
        res = tracker.compare_tonals()
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
    cfg = tyro.cli(TonalConfig)
    eval_graph_search(cfg)