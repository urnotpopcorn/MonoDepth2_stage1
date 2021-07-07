CUDA_VISIBLE_DEVICES=5 \
    python3 train.py \
    --model_name stage12_monoori_woSIG_davis_bear \
    --log_dir log \
    --batch_size 2 \
    --num_workers 2 \
    --log_frequency 50 \
    --learning_rate 1e-5 \
    --load_weights_folder log/stage1_monoori_woSIG_davis_bear/models/weights_99 \
    --instance_pose \
    --fix_pose \
    --fix_depth \
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
    #--disable_pose_invert
    #--predict_delta
    #--load_weights_folder models/mono_640x192 \
    #--project_dir /home/qhhuang/monodepth-project \
