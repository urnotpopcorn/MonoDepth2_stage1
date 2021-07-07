#!/bin/bash
BASEDIR=../MonoDepth2/models
CUDAIDX=1
FLOWDIR=output/flow
OUTDIR=result_eval_flow
MODEL=mono_640x192
EPOCH=0

# step1: compute flow
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow2.py \
    --load_weights_folder ${BASEDIR}/${MODEL}/ \
    --input_dir dataset/data_scene_flow/training/image_2 \
    --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
    --batch_size 1 