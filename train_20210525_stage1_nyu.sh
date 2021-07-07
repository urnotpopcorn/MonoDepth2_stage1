CUDA_VISIBLE_DEVICES=2 \
    python3 train.py \
    --model_name stage1_nyuv2_smooth1e-2 \
    --disparity_smoothness 1e-2 \
    --split nyuv2 --dataset nyuv2 --data_path dataset/NYUv2/train --min_depth 0.1 --max_depth 10 --height 256 --width 320 \
    --log_dir log --png \
    --batch_size 20 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 
