CUDA_VISIBLE_DEVICES=2 \
    python3 train.py \
    --model_name test \
    --log_dir log --png \
    --batch_size 1 \
    --num_workers 1 \
    --log_frequency 1 \
    --learning_rate 1e-5 \
    --load_weights_folder log/stage12_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_roi4e-1_new/models/weights_16 \
    --SIG \
    --instance_pose \
    --instance_motion \
    --instance_motion_v2 \
    --fix_pose \
    --fix_depth \
    --fix_ins_pose \
    --disable_pose_invert \
    --disable_inspose_invert \
    --weight_fg 1 \
    --ext_recept_field \
    --mask_loss_weight 1 \
    --set_y_zero \
    --use_depth_ordering 
    # --predict_img_motion
    # --train_filter_warping_error \
    # --max_speed 0.278
    #--eval_flow_filter_size
    #--disable_pose_invert
    #--predict_delta
    #--load_weights_folder models/mono_640x192 \
    #--project_dir /home/qhhuang/monodepth-project \
