CUDA_VISIBLE_DEVICES=6 \
    python3 train.py \
    --model_name stage12_srcwarp_3 \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --load_weights_folder ../MonoDepth2/models/mono_640x192/ \
    --instance_pose \
    --fix_pose \
    --fix_depth \
    --disable_inspose_invert \
    --weight_fg 1 \
    --ext_recept_field \
    --mask_loss_weight 1 \
    --set_y_zero \
    --use_depth_ordering \
    --roi_diff_thres 0.4 \
    --srcwarp
    
    # --SIG \
    #--disable_pose_invert
    #--predict_delta
    #--load_weights_folder models/mono_640x192 \
    #--project_dir /home/qhhuang/monodepth-project \
