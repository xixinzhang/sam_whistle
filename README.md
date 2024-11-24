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
python sam_whistle/main.py --model sam --spect_cfg.block_multi 10 --batch_size 8 --device cuda:0
python sam_whistle/main.py --model deep --spect_cfg.block_multi 1 --device cuda:1
python sam_whistle/main.py --model fcn_spect --spect_cfg.block_multi 1 --batch_size 8 --device cuda:2
python sam_whistle/main.py --model fcn_encoder --spect_cfg.block_multi 1 --batch_size 8 --device cuda:3
```
inference
```shell
```
## evaluation
1. pixel-wise
   ```shell
   python sam_whistle/evaluate/eval_conf.py --eval_single --model deep --log_dir logs/10-06-2024_15-20-41_pu --min_thre 0.01 --max_thre 0.95
   python sam_whistle/evaluate/eval_conf.py --eval_single --model sam --log_dir logs/11-16-2024_23-45-27
   python sam_whistle/evaluate/evaluate.py --no_single_eval
   ```
2. tone-wise
   1. extract tonal
        ```shell
        python sam_whistle/evaluate/tonal_extraction/tonal_tracker.py --spect_cfg.crop
        ```
   2. evaluate
      ```shell
      python sam_whistle/evaluate/eval_tonal.py --use_conf --log_dir logs/11-16-2024_23-45-27
      ```
## Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)