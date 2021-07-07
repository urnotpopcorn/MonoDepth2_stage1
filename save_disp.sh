#!/bin/bash
BASEDIR=~/xzwu/Code/MonoDepth2_stage1/log
# BASEDIR=../MonoDepth2/models/mono_640x192
MODEL=
EPOCH=10
CUDAIDX=0
IMGIDX=2

CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 save_disp.py \
    --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
    --input_dir dataset/data_scene_flow/training/image_${IMGIDX} \
    --output_dir output/disp/${MODEL}/weights_${EPOCH}/image_${IMGIDX} 
    # --load_weights_folder ${BASEDIR} \
    # --SIG \
    # --instance_pose \
    # --batch_size 1 \
    # --disable_pose_invert \
    # --disable_inspose_invert \
    # --ext_recept_field \
    # --use_depth_ordering \
    # --set_y_zero \
    # --eval_flow_filter_warping_error