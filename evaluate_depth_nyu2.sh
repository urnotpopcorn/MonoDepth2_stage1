#!/bin/bash
BASEDIR=log
OUTDIR=result_eval_depth_nyu
CUDA_IDX=2

MODEL=../../Pretrained/MonoDepth2/mono_640x192
OUTPATH=${OUTDIR}/mono_640x192/weights_0.log
mkdir ${OUTDIR}/mono_640x192

CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_nyu.py \
     --load_weights_folder ${MODEL} \
     --data_path dataset \
     --min_depth 0.1 --max_depth 10 #> ${OUTPATH}
     # --height 256 --width 320

python3 result_eval_depth_nyu/print_value.py result_eval_depth_nyu/mono_640x192
