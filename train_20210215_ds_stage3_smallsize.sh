CUDA_VISIBLE_DEVICES=1 \
    python3 train.py \
    --model_name stage123v2_monoori_woSIG_ds_640192 \
    --log_dir log \
    --batch_size 8 \
    --num_workers 8 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --height 192 \
    --width 640 \
    --dataset drivingstereo_eigen \
    --data_path dataset/drivingstereo \
    --split drivingstereo_eigen \
    --load_weights_folder log/stage12_monoori_woSIG_ds_640192/models/weights_18 \
    --instance_pose \
    --instance_motion \
    --instance_motion_v2 \
    --fix_pose \
    --fix_depth \
    --fix_ins_pose \
    --disable_inspose_invert \
    --weight_fg 1 \
    --ext_recept_field \
    --mask_loss_weight 1 \
    --set_y_zero \
    --use_depth_ordering 
    # --SIG \
