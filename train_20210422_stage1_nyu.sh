CUDA_VISIBLE_DEVICES=3 \
    python3 train.py \
    --model_name stage1_woPool_nyuv2 \
    --split nyuv2 --dataset nyuv2 --data_path dataset --min_depth 0.1 --max_depth 10 --height 320 --width 320 \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 
