#!/bin/bash
MODEL=stage12_monoori_woSIG_davis_bear
EPOCH=99
CUDAIDX=7
# MODEL=mono_640x192
# EPOCH=0

CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 create_video_DAVIS.py \
    --input_dir output/vis_video_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution_selected/${MODEL}/weights_${EPOCH}/ \
    --output_dir output/vis_video_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution_selected/${MODEL}/weights_${EPOCH}/ \
    --rgb_input_dir DAVIS_2017/DAVIS-2017-Unsupervised-trainval-Full-Resolution/JPEGImages/Full-Resolution 
    # --rgb_input_dir DAVIS_2017/DAVIS-2019-Unsupervised-test-dev-Full-Resolution_selected/JPEGImages/Full-Resolution 

# CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 create_video_DAVIS.py \
#     --input_dir output/vis_video_DAVIS/DAVIS-2017-Unsupervised-trainval-Full-Resolution/${MODEL}/weights_${EPOCH}/ \
#     --output_dir output/vis_video_DAVIS/DAVIS-2017-Unsupervised-trainval-Full-Resolution/${MODEL}/weights_${EPOCH}/ \
#     --rgb_input_dir DAVIS_2017/DAVIS-2017-Unsupervised-trainval-Full-Resolution/JPEGImages/Full-Resolution 
