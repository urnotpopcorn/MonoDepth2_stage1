CUDA_VISIBLE_DEVICES=0 \
    python3 train.py \
    --model_name stage1_dp_ninit \
    --depth_decoder_normal_init \
    --pose_decoder_normal_init \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --num_epochs 60
    # --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
