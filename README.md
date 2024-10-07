## Segment any whistle using foundational model SAM

## Install
`conda install ffmpeg`


train
```shell
python sam_whistle/main.py --no_sam_decoder --loss_fn bce_logits
```
evaluate
```shell
python sam_whistle/evaluate/evaluate.py --no_sam_decoder --visualize_eval --save_path /home/asher/Desktop/projects/sam_whistle/logs/09-23-2024_10-00-40
```

## Acknowledgement
- [SAM](https://github.com/facebookresearch/segment-anything)