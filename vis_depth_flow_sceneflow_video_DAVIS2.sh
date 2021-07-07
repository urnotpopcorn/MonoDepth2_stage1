#!/bin/bash
BASEDIR=../MonoDepth2/models
CUDAIDX=0
# FLOWDIR=output/vis_video_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution
# FLOWDIR=output/vis_video_DAVIS/DAVIS-2017-Unsupervised-trainval-Full-Resolution
FLOWDIR=output/vis_video_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution_selected
MODEL=mono_640x192
EPOCH=0

mkdir ${FLOWDIR}/${MODEL}
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 vis_depth_flow_sceneflow_video_DAVIS.py \
    --load_weights_folder ${BASEDIR}/${MODEL} \
    --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
    --batch_size 1 