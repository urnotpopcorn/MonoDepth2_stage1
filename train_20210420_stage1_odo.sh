CUDA_VISIBLE_DEVICES=2 \
    python3 train.py \
    --model_name stage1_woPool_odo \
    --split odom --dataset kitti_odom --data_path dataset/kitti_odometry \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-4 
