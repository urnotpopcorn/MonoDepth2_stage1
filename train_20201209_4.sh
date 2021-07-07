CUDA_VISIBLE_DEVICES=4 \
    python3 train.py \
    --model_name stage1_rpose_320x1024 \
    --log_dir log --png --height 320 --width 1024 \
    --batch_size 4 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --SIG \
    --disable_pose_invert
