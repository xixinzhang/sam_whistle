import json
import os

from sam_whistle.evaluate.tonal_extraction.tonal_tracker import (Config,
                                                                 TonalTracker)

cfg = Config()

def get_detections(cfg):
    # TODO: train + test
    stems = json.load(open(os.path.join(cfg.root_dir, cfg.meta_file)))['test']
    stems = stems[:1]
    trackers = {}
    for stem in stems:
        tracker = TonalTracker(cfg, stem)
        tracker.sam_inference()
        trackers[stem]= tracker
        tracker.build_graph()
        tonals = tracker.get_tonals()
        for tonal in tonals:  # [[T, F]]
            print(tonal.min(axis=0))
            print(tonal.max(axis=0))

def dt_to_coco():
    pass

def match_data():
    pass

if __name__ == "__main__":
    get_detections(cfg)