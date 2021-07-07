CUDA_VISIBLE_DEVICES=1 \
    python3 train.py \
    --model_name stage12_monoori_woSIG_ds_640192 \
    --log_dir log \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --height 192 \
    --width 640 \
    --dataset drivingstereo_eigen \
    --data_path dataset/drivingstereo \
    --split drivingstereo_eigen \
    --load_weights_folder ../MonoDepth2_stage1/log/stage1_monoori_woSIG_ds_640192/models/weights_11 \
    --instance_pose \
    --fix_pose \
    --fix_depth \
    --disable_inspose_invert \
    --weight_fg 1 \
    --ext_recept_field \
    --mask_loss_weight 1 \
    --set_y_zero \
    --use_depth_ordering \
    --roi_diff_thres 0.4
    #--SIG \
    #--disable_pose_invert
    #--predict_delta
    #--load_weights_folder models/mono_640x192 \
    #--project_dir /home/qhhuang/monodepth-project \
