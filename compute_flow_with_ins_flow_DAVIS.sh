#!/bin/bash
BASEDIR=log
CUDAIDX=0
FLOWDIR=output/flow_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution
MODEL=stage123v2_monoori_woSIG_srcsample2_roi_y0
EPOCH=11

CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow_DAVIS.py \
    --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
    --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
    --batch_size 1 \
    --instance_pose \
    --instance_motion \
    --instance_motion_v2 \
    --disable_pose_invert \
    --disable_inspose_invert \
    --ext_recept_field \
    --set_y_zero \
    --use_depth_ordering \
    --eval_flow_mask_outer \
    --filter_mask \
    --eval_flow_filter_warping_error_stage2 \
    --eval_flow_filter_warping_error_stage3 \
    --roi_diff_thres 0.2 
