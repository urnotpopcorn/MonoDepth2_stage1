CUDA_VISIBLE_DEVICES=7 \
    python3 train.py \
    --model_name stage123v2_monoori_woSIG_srcsample \
    --log_dir log --png \
    --batch_size 12 \
    --num_workers 12 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --load_weights_folder log/stage12_srcwarp_2_bmm/models/weights_16 \
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