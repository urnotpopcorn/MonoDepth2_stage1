CUDA_VISIBLE_DEVICES=1 \
    python3 train.py \
    --model_name stage123v2_monoori_woSIG_srcsample2_roi_y0_ds_depth \
    --log_dir log \
    --batch_size 4 \
    --num_workers 4 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --height 288 \
    --width 640 \
    --dataset drivingstereo_eigen \
    --data_path dataset/drivingstereo \
    --split drivingstereo_eigen \
    --load_weights_folder log/stage123v2_monoori_woSIG_srcsample2_roi_y0_ds/models/weights_8 \
    --instance_pose \
    --instance_motion \
    --instance_motion_v2 \
    --fix_pose \
    --fix_ins_pose \
    --fix_ins_motion \
    --disable_inspose_invert \
    --ext_recept_field \
    --filter_mask \
    --eval_flow_filter_warping_error_stage2 \
    --eval_flow_filter_warping_error_stage3 \
    --roi_diff_thres 0.2 \
    --set_y_zero \
    --use_depth_ordering 
    # --eval_flow_mask_outer \
    #--weight_fg 1 \
    #--mask_loss_weight 1 \
    #--fix_depth \
    # --SIG \
