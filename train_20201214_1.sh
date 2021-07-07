CUDA_VISIBLE_DEVICES=1 \
    python3 train.py \
    --model_name stage1_rpose_max40 \
    --log_dir log --png --height 192 --width 640 --num_epochs 40 \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --SIG \
    --disable_pose_invert 
