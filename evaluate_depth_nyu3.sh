#!/bin/bash
BASEDIR=log
OUTDIR=result_eval_depth_nyu
CUDA_IDX=1

# for MODEL in stage1_nyuv2rec_smooth1e-2
# do
# 	mkdir ${OUTDIR}/${MODEL}
# 	for EPOCH in {28..26..-1}
# 	do
# 	    OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
# 	    echo ${OUTPATH}
# 	    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_nyu.py \
# 		--load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
# 		--data_path dataset --min_depth 0.1 --max_depth 10 --height 256 --width 320 > ${OUTPATH}
# 	done
# 	python3 result_eval_depth_nyu/print_value.py result_eval_depth_nyu/${MODEL}
# done

for MODEL in stage1_nyuv2rec_smooth1e-1
do
	mkdir ${OUTDIR}/${MODEL}
	for EPOCH in {32..0..-1}
	do
	    OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
	    echo ${OUTPATH}
	    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_nyu.py \
		--load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
		--data_path dataset --min_depth 0.1 --max_depth 10 --height 256 --width 320 > ${OUTPATH}
	done
	python3 result_eval_depth_nyu/print_value.py result_eval_depth_nyu/${MODEL}
done
