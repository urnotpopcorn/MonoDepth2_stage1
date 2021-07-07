#!/bin/bash
OCC=occ # NEED change compute_3d_flow_with_ins_flow4.py at line 1415
CUDAIDX=7
BASEDIR=../MonoDepth2/models
FLOWDIR=output/sceneflow
MODEL=mono_640x192
OUTDIR=${FLOWDIR}/${OCC}/${MODEL}/weights_${EPOCH}
mkdir -p ${OUTDIR}
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_3d_flow_with_ins_flow4.py \
    --load_weights_folder ${BASEDIR}/${MODEL} \
    --input_dir dataset/data_scene_flow/training/image_2 \
    --output_dir ${OUTDIR} \
    --batch_size 1 

BASEDIR=log
FLOWDIR=output/sceneflow_tgt_wflow4
MODEL=stage12_srcwarp_2_bmm
EPOCH=18
OUTDIR=${FLOWDIR}/${OCC}/${MODEL}/weights_${EPOCH}
mkdir -p ${OUTDIR}
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_3d_flow_with_ins_flow4.py \
    --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
    --input_dir dataset/data_scene_flow/training/image_2 \
    --output_dir ${OUTDIR} \
    --batch_size 1 \
    --instance_pose \
    --disable_pose_invert \
    --disable_inspose_invert \
    --ext_recept_field \
    --filter_mask \
    --eval_flow_filter_warping_error_stage2 \
    --roi_diff_thres 0.2 \
    --set_y_zero \
    --use_depth_ordering
    # --eval_flow_mask_outer \

MODEL=stage123v2_monoori_woSIG_srcsample2_roi_y0
EPOCH=11
OUTDIR=${FLOWDIR}/${OCC}/${MODEL}/weights_${EPOCH}
mkdir -p ${OUTDIR}
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_3d_flow_with_ins_flow4.py \
    --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
    --input_dir dataset/data_scene_flow/training/image_2 \
    --output_dir ${OUTDIR} \
    --batch_size 1 \
    --instance_pose \
    --instance_motion \
    --instance_motion_v2 \
    --disable_pose_invert \
    --disable_inspose_invert \
    --ext_recept_field \
    --filter_mask \
    --eval_flow_filter_warping_error_stage2 \
    --eval_flow_filter_warping_error_stage3 \
    --roi_diff_thres 0.2 \
    --set_y_zero \
    --use_depth_ordering
    # --eval_flow_mask_outer \