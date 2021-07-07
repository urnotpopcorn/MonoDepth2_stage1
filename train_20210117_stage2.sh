CUDA_VISIBLE_DEVICES=7 \
    python3 train.py \
    --model_name stage12_monoori_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_roi4e-1 \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
    --SIG \
    --instance_pose \
    --fix_pose \
    --fix_depth \
    --disable_pose_invert \
    --disable_inspose_invert \
    --weight_fg 1 \
    --ext_recept_field \
    --mask_loss_weight 1 \
    --set_y_zero \
    --use_depth_ordering \
    --roi_diff_thres 0.4

    #--disable_pose_invert
    #--predict_delta
    #--load_weights_folder models/mono_640x192 \