#!/bin/bash
BASEDIR=../MonoDepth2_polar/log2
OUTDIR=../MonoDepth2_polar/result_eval_depth_nyu
CUDA_IDX=2

for MODEL in stage1_nyuv2_smooth1e-2_lpips1e-2 stage1_nyuv2_smooth1e-2_lpips1e-1
do
	mkdir ${OUTDIR}/${MODEL}
	for EPOCH in {59..0..-1}
	    do
	    OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
	    echo ${OUTPATH}
	    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_nyu.py \
		--load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
		--data_path dataset --min_depth 0.1 --max_depth 10 --height 256 --width 320 > ${OUTPATH}
		# --height 256 --width 320
	    done
done
