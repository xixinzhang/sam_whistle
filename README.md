## Segment any whistle using foundational model SAM

## Install
`conda install ffmpeg`


train
```shell
 python sam_whistle/main.py --model fcn_encoder --exp_name fcn_encoder
```
evaluate
```shell
python sam_whistle/evaluate/evaluate.py --model fcn_encoder --save_path /home/asher/Desktop/projects/sam_whistle/logs/09-23-2024_10-00-40  --single_eval --visualize_eval
python sam_whistle/evaluate/evaluate.py --no_single_eval
```

## Acknowledgement
- [SAM](https://github.com/facebookresearch/segment-anything)