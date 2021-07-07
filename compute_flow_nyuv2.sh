#!/bin/bash
BASEDIR=log
CUDAIDX=7
FLOWDIR=output/flow4
OUTDIR=result_eval_flow4

for MODEL in 'stage1_nyuv2rec_smooth1e-2'
do
    mkdir ${OUTDIR}/${MODEL}
    for EPOCH in {22..22..-1}
    do 
        # step1: compute flow
        CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_nyuv2.py \
            --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
            --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
            --batch_size 1 
    done
done
