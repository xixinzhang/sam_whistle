## Segment any whistle using foundational model SAM

## Install
```
conda create -n sam_whistle python=3.12
conda activate sam_whistle
pip install -e .
git submodule add --force https://github.com/facebookresearch/segment-anything
pip install -e ./segment-anything/
```
`conda install ffmpeg`


## Data processing
no skeleton, no crop
```shell
python sam_whistle/datasets/dataset.py --preprocess --all_data --spect_cfg.block_multi 1
```

## Training & Inference
training
```shell
python sam_whistle/main.py --model sam --spect_cfg.block_multi 10 --batch_size 2 --device cuda:0
python sam_whistle/main.py --model deep --spect_cfg.block_multi 1 --device cuda:0
python sam_whistle/main.py --model fcn_spect --spect_cfg.block_multi 1 --batch_size 2 --device cuda:0
python sam_whistle/main.py --model fcn_encoder --spect_cfg.block_multi 1 --batch_size 2 --device cuda:0
```
inference
```shell
```
## evaluation
1. pixel-level
   ```shell
   python sam_whistle/evaluate/eval_conf.py --model sam --log_dir logs/11-23-2024_15-19-19-sam
   python sam_whistle/evaluate/eval_conf.py --model deep --log_dir logs/11-23-2024_15-27-33-deep_whistle
   python sam_whistle/evaluate/eval_conf.py --model fcn_spect --log_dir logs/11-23-2024_15-39-59-fcn_spect
   python sam_whistle/evaluate/eval_conf.py --model fcn_encoder --log_dir logs/11-24-2024_03-02-50-fcn_encoder
   ```
2. tonal-wise
   1. extract tonal
        ```shell
        python sam_whistle/evaluate/tonal_extraction/tonal_tracker.py
        ```
   2. evaluate
      ```shell
      python sam_whistle/evaluate/eval_tonal.py --use_conf --log_dir logs/11-16-2024_09-28-58  --model sam # 1e-10
      ```
## Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)