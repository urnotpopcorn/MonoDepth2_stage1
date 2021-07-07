CUDA_VISIBLE_DEVICES=7 \
    python3 train.py \
    --model_name stage123_monoori_woSIG_davis_bear \
    --log_dir log \
    --batch_size 2 \
    --num_workers 2 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --load_weights_folder log/stage12_monoori_woSIG_davis_bear/models/weights_99 \
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
    --use_depth_ordering \
    --roi_diff_thres 0 \
    --dataset davis \
    --data_path DAVIS_2017 \
    --split davis \
    --num_epochs 100
    # --SIG \
