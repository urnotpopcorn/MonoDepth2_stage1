
#!/bin/bash
BASEDIR=log
CUDAIDX=1
FLOWDIR=output/flow_ds

CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow3.py \
    --load_weights_folder log/stage1_rpose/models/weights_15 \
    --output_dir ${FLOWDIR}/stage1_rpose/weights_15 \
    --batch_size 1 \
    --SIG \
    --instance_pose \
    --disable_pose_invert \
    --disable_inspose_invert \
    --ext_recept_field \
    --dataset 'drivingstereo_eigen'
    # --roi_diff_thres 0.4 