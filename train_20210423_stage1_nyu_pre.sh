CUDA_VISIBLE_DEVICES=1 \
    python3 train.py \
    --model_name stage1_nyuv2_pre \
    --split nyuv2 --dataset nyuv2 --data_path dataset --min_depth 0.1 --max_depth 10 \
    --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 
