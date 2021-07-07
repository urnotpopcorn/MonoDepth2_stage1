#!/bin/bash
BASEDIR=log
CUDAIDX=0
FLOWDIR=output/flow_ds
OUTDIR=result_eval_flow_ds

for MODEL in 'stage123v2_monoori_woSIG'
do
    mkdir ${OUTDIR}/${MODEL}
    for EPOCH in {15..15..-1}
    do
        # step1: compute flow
        CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow2.py \
            --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
            --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
            --batch_size 1 \
            --instance_pose \
            --instance_motion \
            --instance_motion_v2 \
            --disable_pose_invert \
            --disable_inspose_invert \
            --ext_recept_field \
            --eval_flow_mask_outer \
            --filter_mask \
            --eval_flow_filter_warping_error_stage2 \
            --eval_flow_filter_warping_error_stage3 \
            --roi_diff_thres 0.2 \
            --set_y_zero \
            --use_depth_ordering \
            --dataset drivingstereo_eigen \
            --height 288 \
            --width 640
            # --input_dir dataset/data_scene_flow/training/image_2 \
            # --SIG \
    done
done
