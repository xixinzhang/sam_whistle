## Segment any whistle using foundational model SAM

## Install
`conda install ffmpeg`


## Data processing
no skeleton, no crop
```shell
 python sam_whistle/datasets/dataset.py --preprocess
```

## Training & Evaluation
train w/o sam decoder
```shell
 python sam_whistle/main.py --model sam 

```
evaluate
```shell
python sam_whistle/evaluate/evaluate.py --model sam --log_dir logs/09-23-2024_10-00-40  --single_eval --visualize_eval
python sam_whistle/evaluate/evaluate.py --no_single_eval
```
extract tonal
```shell
python sam_whistle/evaluate/tonal_extraction/graph_search.py --spect_config.crop
```
## Acknowledgement
- [SAM](https://github.com/facebookresearch/segment-anything)