## SAM-Whistle: Adapting Foundation Models for Automated Dolphin Whistle Detection

## Install
```
conda create -n sam_whistle python=3.12
conda activate sam_whistle
pip install -e .
git submodule add --force https://github.com/facebookresearch/segment-anything
pip install -e ./segment-anything/
```
ffmpeg is required: `conda install ffmpeg`

## Training
Your dataset should be organized in the following structure:
```
data
├── dclde
│   ├── anno # Annotation files
│   ├── audio # Audio recordings
│   ├── meta.json  # data info
└── killer
    └── audio
```

structure of meta.json:
```
{
"train":[stem1, stem2,...],
"test":[stem1, stem2,...]
}
```
You can train the model from scratch on your own dataset using the following command:
```shell
python sam_whistle/main.py --model sam --batch_size 2 --device cuda:0 --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3
```
## Inference
A trained model [checkpoint](https://drive.google.com/drive/folders/1LXkczIaOyIMcu4Zmiamgc6jbtw6rrupC?usp=sharing) for the DCLDE 2011 dataset can be downloaded and used (@0.5) for inference with:
```shell
python sam_whistle/evaluate/eval_tonal.py --use_conf --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/sam --min_thre 0.5 --max_thre 0.5
```

## Evaluation
1. pixel-level
```shell
python sam_whistle/evaluate/eval_conf.py --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/sam
```
2. tonal-wise
```shell
python sam_whistle/evaluate/eval_tonal.py --use_conf --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/sam --min_thre 0.01 --max_thre 0.99 --thre_num 20
```
## Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)
- [DeepWhistle](https://github.com/Paul-LiPu/DeepWhistle)
- [silbido](https://github.com/MarineBioAcousticsRC/silbido)