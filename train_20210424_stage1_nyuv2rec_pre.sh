CUDA_VISIBLE_DEVICES=5 \
    python3 train.py \
    --model_name stage1_nyuv2rec_pre \
    --split nyuv2rec --dataset nyuv2rec --data_path dataset/NYUv2_rectified/train --min_depth 0.1 --max_depth 10 --height 192 --width 640 \
    --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --num_epochs 60
