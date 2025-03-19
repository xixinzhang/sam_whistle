# data processing and visualization
# python sam_whistle/datasets/dataset.py --preprocess --spect_cfg.block_multi 1 --debug --spect_cfg.interp linear --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.kernel_size 3

# train model
python sam_whistle/main.py --model sam --batch_size 2 --device cuda:0 --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3
python sam_whistle/main.py --model deep  --device cuda:0 --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3
python sam_whistle/main.py --model fcn_spect  --batch_size 2 --device cuda:0 --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3
python sam_whistle/main.py --model fcn_encoder  --batch_size 2 --device cuda:0 --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3

# test conf
python sam_whistle/evaluate/eval_conf.py --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-29-sam
python sam_whistle/evaluate/eval_conf.py --model deep --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-24-29-deep_whistle
python sam_whistle/evaluate/eval_conf.py --model fcn_spect --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-52-54-fcn_spect
python sam_whistle/evaluate/eval_conf.py --model fcn_encoder --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-03-fcn_encoder

python sam_whistle/evaluate/eval_conf.py --eval_multiple

# test tonal detection
python sam_whistle/evaluate/eval_tonal.py --use_conf --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-29-sam --min_thre 0.01 --max_thre 0.99 --thre_num 20
python sam_whistle/evaluate/eval_tonal.py --use_conf --model deep --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-24-29-deep_whistle --min_thre 0.01 --max_thre 0.99 --thre_num 20
python sam_whistle/evaluate/eval_tonal.py --use_conf --model fcn_spect --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-52-54-fcn_spect --min_thre 0.01 --max_thre 0.99 --thre_num 20
python sam_whistle/evaluate/eval_tonal.py --use_conf --model fcn_encoder --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-03-fcn_encoder --min_thre 0.01 --max_thre 0.99 --thre_num 20
python sam_whistle/evaluate/eval_tonal.py --model graph_search --spect_cfg.normalize zscore  --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/graph_search --min_thre 9.5 --max_thre 10.5 --thre_num 20

python sam_whistle/evaluate/eval_tonal.py --eval_multiple

# visulize extraction
python sam_whistle/evaluate/eval_tonal.py --use_conf --model sam --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-29-sam --min_thre 0.06939393939393938 --max_thre 0.06939393939393938 --visualize --output_dir outputs/paper/sam
python sam_whistle/evaluate/eval_tonal.py --use_conf --model deep --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-24-29-deep_whistle --min_thre 0.45545454545454545 --max_thre 0.45545454545454545 --visualize --output_dir outputs/paper/deep
python sam_whistle/evaluate/eval_tonal.py --use_conf --model fcn_spect --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-52-54-fcn_spect --min_thre 0.2673737373737374 --max_thre 0.2673737373737374 --visualize --output_dir outputs/paper/fcn_spect
python sam_whistle/evaluate/eval_tonal.py --use_conf --model fcn_encoder --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/01-21-2025_19-51-03-fcn_encoder --min_thre 0.059494949494949496 --max_thre 0.059494949494949496 --visualize --output_dir outputs/paper/fcn_encoder
python sam_whistle/evaluate/eval_tonal.py --model graph_search --spect_cfg.normalize zscore  --spect_cfg.interp linear --spect_cfg.kernel_size 3 --log_dir logs/graph_search --min_thre 9.681818181818182 --max_thre 9.681818181818182 --visualize --output_dir outputs/paper/graph_search


# coco
# sam1
python sam_whistle/main.py --model sam_coco --batch_size 2 --device cuda:0
python sam_whistle/evaluate/eval_coco.py --log-dir logs/03-07-2025_21-22-39-sam_coco --model_name sam
# sam2
python sam_whistle/evaluate/eval_coco.py --log-dir logs/03-14-2025_17-58-44-sam2_coco --model_name sam2
python sam_whistle/main.py --model sam2_coco --batch_size 2 --device cuda:0
