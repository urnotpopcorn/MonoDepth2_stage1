CUDA_VISIBLE_DEVICES=6 \
    python3 train.py \
    --model_name stage123v2_monoori_woSIG_srcsample2_roi_maskloss_depth \
    --log_dir log --png \
    --batch_size 6 \
    --num_workers 6 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --load_weights_folder log/stage123v2_monoori_woSIG_srcsample2_roi/models/weights_12 \
    --instance_pose \
    --instance_motion \
    --instance_motion_v2 \
    --fix_pose \
    --fix_ins_pose \
    --fix_ins_motion \
    --disable_inspose_invert \
    --ext_recept_field \
    --eval_flow_mask_outer \
    --filter_mask \
    --eval_flow_filter_warping_error_stage2 \
    --eval_flow_filter_warping_error_stage3 \
    --roi_diff_thres 0.2 \
    --set_y_zero \
    --use_depth_ordering \
    --mask_loss_weight 1
    # --weight_fg 1 \
    # --mask_loss_weight 1 
    #--fix_depth \
    # --SIG \
