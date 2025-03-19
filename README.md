## SAM-Whistle: Adapting Foundation Models for Automated Dolphin Whistle Detection

## Install
```
conda create -n sam_whistle python=3.12
conda activate sam_whistle
pip install -e .
git submodule add --force https://github.com/facebookresearch/segment-anything
pip install -e ./segment-anything/
```
`conda install ffmpeg`


## Training & Inference
training
```shell
python sam_whistle/main.py --model sam --batch_size 2 --device cuda:0 --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3
```
inference
```shell
```
## evaluation
1. pixel-level
```shell
python sam_whistle/evaluate/eval_conf.py --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-29-sam
```
2. tonal-wise
```shell
python sam_whistle/evaluate/eval_tonal.py --use_conf --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-29-sam --min_thre 0.01 --max_thre 0.99 --thre_num 20
```

# COCO format
## Training
```shell
python sam_whistle/main.py --model sam_coco --batch_size 2 --device cuda:0

```

## Evaluation
```shell
python sam_whistle/evaluate/eval_coco.py --log-dir logs/03-07-2025_21-22-39-sam_coco

```
## Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [DeepWhistle](https://github.com/Paul-LiPu/DeepWhistle)
- [silbido](https://github.com/MarineBioAcousticsRC/silbido)