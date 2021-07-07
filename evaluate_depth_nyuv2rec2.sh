#!/bin/bash
BASEDIR=log
OUTDIR=result_eval_depth_nyuv2rec
CUDA_IDX=6

MODEL=stage1_nyuv2rec_rpose
mkdir ${OUTDIR}/${MODEL}
for EPOCH in {59..6..-1}
    do
    OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
    echo ${OUTPATH}
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_nyuv2rec.py \
        --data_path dataset/NYUv2_rectified/test --dataset nyuv2rec --eval_split nyuv2rec \
        --min_depth 0.1 --max_depth 10 --width 320 --height 256 \
        --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
        --batch_size 4 --num_workers 4 --eval_mono --png > ${OUTPATH}
    done

python3 result_eval_depth_nyuv2rec/print_value.py result_eval_depth_nyuv2rec/${MODEL}

# MODEL=../../Pretrained/MonoDepth2/mono_640x192
# OUTPATH=${OUTDIR}/mono_640x192/weights_0.log
# CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_nyuv2rec.py \
#     --data_path dataset/NYUv2_rectified/test --dataset nyuv2rec --eval_split nyuv2rec \
#     --min_depth 0.1 --max_depth 10 \
#     --load_weights_folder ${MODEL} \
#     --batch_size 8 --num_workers 8 --eval_mono --png > ${OUTPATH}
