import tyro
import json
import os

from sam_whistle.config import TonalConfig
from sam_whistle.evaluate.tonal_extraction.tonal_tracker import TonalTracker


def eval_graph_search(cfg: TonalConfig):
    stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))['train']
    for stem in stems:
        tracker = TonalTracker(cfg, stem)
        tracker.build_graph()
        tonals = tracker.get_tonals()
        print(stem)
        print(len(tonals), tracker.discarded_count)
        # print(tonals[0])
        # print(tonals[-1])

        # TODO: evaluation code


if __name__ == "__main__":
    cfg = tyro.cli(TonalConfig)
    eval_graph_search(cfg)