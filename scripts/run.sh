# data processing and visualization
# python sam_whistle/datasets/dataset.py --preprocess --spect_cfg.block_multi 1 --debug --spect_cfg.interp linear --spect_cfg.normalize zscore --spect_cfg.no_center --spect_cfg.kernel_size 2

# train model
python sam_whistle/main.py --model sam --batch_size 2 --device cuda:0 --spect_cfg.block_multi 5 --spect_cfg.normalize fixed_minmax --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 2

# test conf
# python sam_whistle/evaluate/eval_conf.py --model sam --spect_cfg.normalize fixed_minmax  --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 2 --log_dir logs/12-15-2024_06-13-53-fixed_minmax_no_trans

# test tonal detection
# python sam_whistle/evaluate/eval_tonal.py --use_conf --model sam --spect_cfg.normalize fixed_minmax --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 2 --log_dir logs/12-15-2024_06-13-53-fixed_minmax_no_trans

python sam_whistle/main.py --model deep --spect_cfg.block_multi 1 --device cuda:0 --spect_cfg.normalize fixed_minmax --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 2
python sam_whistle/main.py --model fcn_spect --spect_cfg.block_multi 1 --batch_size 2 --device cuda:0 --spect_cfg.normalize fixed_minmax --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 2
python sam_whistle/main.py --model fcn_encoder --spect_cfg.block_multi 1 --batch_size 2 --device cuda:0 --spect_cfg.normalize fixed_minmax --spect_cfg.no_center --spect_cfg.interp linear --spect_cfg.kernel_size 2