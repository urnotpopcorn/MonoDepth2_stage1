#!/bin/bash
BASEDIR=log
OUTDIR=result_eval_depth
CUDA_IDX=7

MODEL=stage1_monoori_woSIG_vkitti_woweather_max40
mkdir ${OUTDIR}/${MODEL}
for EPOCH in {39..27..-1}
    do
    OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
    echo ${OUTPATH}
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth.py \
        --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
        --batch_size 8 --num_workers 8 --eval_mono --png > ${OUTPATH}
        #--batch_size 8 --num_workers 8 --eval_mono --png --SIG --height 320 --width 1024 > ${OUTPATH}
    done

python3 result_eval_depth/print_value.py result_eval_depth/${MODEL}

# MODEL=../../Pretrained/MonoDepth2/mono_640x192
# OUTPATH=${OUTDIR}/mono_640x192/weights_0.log
# CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth.py \
#     --load_weights_folder ${MODEL} \
#     --batch_size 8 --num_workers 8 --eval_mono --png > ${OUTPATH}
