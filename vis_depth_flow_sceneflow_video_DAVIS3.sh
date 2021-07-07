#!/bin/bash
BASEDIR=log
CUDAIDX=7
FLOWDIR=output/vis_video_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution_selected
# FLOWDIR=output/vis_video_DAVIS/DAVIS-2017-Unsupervised-trainval-Full-Resolution
 
# MODEL=stage1_monoori_woSIG_davis_bike-packing
# EPOCH=99
# mkdir ${FLOWDIR}/${MODEL}
# CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 vis_depth_flow_sceneflow_video_DAVIS.py \
#     --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
#     --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
#     --batch_size 1 

# MODEL=stage12_monoori_woSIG_davis_bear
# EPOCH=99
# CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 vis_depth_flow_sceneflow_video_DAVIS.py \
#     --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
#     --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
#     --batch_size 1 \
#     --instance_pose \
#     --disable_pose_invert \
#     --disable_inspose_invert \
#     --ext_recept_field \
#     --filter_mask \
#     --set_y_zero \
#     --use_depth_ordering
#     # --roi_diff_thres 0.2 \
#     # --eval_flow_filter_warping_error_stage2 \

MODEL=stage123_monoori_woSIG_davis_bear
EPOCH=99
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 vis_depth_flow_sceneflow_video_DAVIS.py \
    --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
    --input_dir dataset/data_scene_flow/training/image_2 \
    --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
    --batch_size 1 \
    --instance_pose \
    --instance_motion \
    --instance_motion_v2 \
    --disable_pose_invert \
    --disable_inspose_invert \
    --ext_recept_field \
    --filter_mask \
    --eval_flow_filter_warping_error_stage3 \
    --set_y_zero \
    --use_depth_ordering
    # --eval_flow_filter_warping_error_stage2 \
    # --roi_diff_thres 0.2 \
