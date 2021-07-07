CUDA_VISIBLE_DEVICES=3 \
    python3 train.py \
    --model_name stage1_rpose_320x1024_max40_new \
    --log_dir log --png --height 320 --width 1024 --num_epochs 40 \
    --batch_size 4 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --SIG \
    --disable_pose_invert 
