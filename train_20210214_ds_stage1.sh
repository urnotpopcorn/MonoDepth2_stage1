CUDA_VISIBLE_DEVICES=5 \
    python3 train.py \
    --model_name stage1_monoori_woSIG_ds \
    --log_dir log \
    --batch_size 10 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 \
    --height 288 \
    --width 640 \
    --dataset drivingstereo_eigen \
    --data_path dataset/drivingstereo \
    --split drivingstereo_eigen 
