#!/bin/bash
CUDAIDX=7

FLOWDIR=output/sceneflow
MODEL=geonet

mkdir ${FLOWDIR}/${MODEL}
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_3d_flow_geonet.py \
    --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
    --height 128 --width 416 \
    --batch_size 1 
