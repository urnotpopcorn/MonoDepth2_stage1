CUDA_VISIBLE_DEVICES=5 \
    python3 train.py \
    --model_name stage1_monoori_woSIG_davis_bmx-bumps_inv \
    --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
    --log_dir log \
    --batch_size 2 \
    --num_workers 2 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --height 192 \
    --width 640 \
    --dataset davis \
    --data_path DAVIS_2017 \
    --split davis \
    --num_epochs 100 \
    --disable_pose_invert
