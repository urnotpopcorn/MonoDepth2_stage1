#!/bin/bash
BASEDIR=~/xzwu/Code/MonoDepth2_stage1/log
MODEL=stage1_monoori_woSIG_vkitti_wweather_max40
EPOCH=22
CUDAIDX=1

CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 save_disp_VKITTI.py \
    --load_weights_folder ../MonoDepth2/models/mono_640x192 \
    --input_dir dataset/VKitti \
    --output_dir output/disp_VKitti/mono_640x192/weights_0/ \
    --batch_size 1 

# CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 save_disp_VKITTI.py \
#     --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
#     --input_dir dataset/VKitti \
#     --output_dir output/disp_VKitti/${MODEL}/weights_${EPOCH}/ \
#     --batch_size 1 
    
    # --SIG \
    # --instance_pose \
    # --batch_size 1 \
    # --disable_pose_invert \
    # --disable_inspose_invert \
    # --ext_recept_field \
    # --use_depth_ordering \
    # --set_y_zero \
    # --eval_flow_filter_warping_error

    # --batch_size 1 \
    # --SIG \
    # --instance_pose \
    # --instance_motion \
    # --instance_motion_v2 \
    # --disable_pose_invert \
    # --disable_inspose_invert \
    # --ext_recept_field \
    # --set_y_zero \
    # --use_depth_ordering \
    # --eval_flow_mask_outer \
    # --filter_mask \
    # --eval_flow_filter_warping_error_stage2 \
    # --eval_flow_filter_warping_error_stage3 \
    # --roi_diff_thres 0.2 \
