## Segment any whistle using foundational model SAM

## Install
`conda install ffmpeg`


## Data processing
no skeleton, no crop
```shell
 python sam_whistle/datasets/dataset.py --preprocess
```

## Training & Inference
training
```shell
 python sam_whistle/main.py --model sam 

```
inference
```shell
```
## evaluation
1. pixel-wise
```shell
python sam_whistle/evaluate/evaluate.py --model sam --log_dir logs/09-23-2024_10-00-40  --single_eval --visualize_eval
python sam_whistle/evaluate/evaluate.py --no_single_eval
```
2. tone-wise
   1. extract tonal
        ```shell
        python sam_whistle/evaluate/tonal_extraction/tonal_tracker.py --spect_cfg.crop
        ```
   2. evaluate
   
## Acknowledgement
- [segment-anything](https://github.com/facebookresearch/segment-anything)