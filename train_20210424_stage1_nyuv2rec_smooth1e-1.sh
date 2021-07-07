CUDA_VISIBLE_DEVICES=3 \
    python3 train.py \
    --model_name stage1_nyuv2rec_smooth1e-1 \
    --split nyuv2rec --dataset nyuv2rec --data_path dataset/NYUv2_rectified/train --min_depth 0.1 --max_depth 10 --height 256 --width 320 \
    --disparity_smoothness 1e-1 \
    --log_dir log --png \
    --batch_size 20 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --num_epochs 60
    # --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
