CUDA_VISIBLE_DEVICES=4 \
    python3 train.py \
    --model_name stage123v2_monoori_woSIG_srcsample_wofgmask2 \
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
    --ext_recept_field \
    --set_y_zero \
    --use_depth_ordering 
    # --SIG \
