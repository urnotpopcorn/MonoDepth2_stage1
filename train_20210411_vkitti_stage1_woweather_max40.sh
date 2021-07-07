CUDA_VISIBLE_DEVICES=0 \
    python3 train.py \
    --model_name stage1_monoori_woSIG_vkitti_woweather_max40 \
    --log_dir log \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --height 192 \
    --width 640 \
    --dataset vkitti \
    --data_path dataset/VKitti \
    --split vkitti \
    --num_epochs 40
    # --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
