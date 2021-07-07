#!/bin/bash
BASEDIR=log
OUTDIR=result_eval_depth
CUDA_IDX=7


# MODEL=stage12fd_recept20
# MODEL=stage12fd_fg1_recept20_maskloss1
# mkdir ${OUTDIR}/${MODEL}
# for EPOCH in {7..7..-1}
#     do
#     OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
#     echo ${OUTPATH}
#     CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth.py \
#         --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
#         --batch_size 8 --num_workers 4 --eval_mono --png --SIG > ${OUTPATH}
#     done

#MODEL=stage123fd_recept20_new_20201117
#MODEL=stage1_rpose_ds
# MODEL=stage12fd_fg1_recept20_maskloss1
#MODEL=mono2_ds
#MODEL=stage123v2_monoori_woSIG_srcsample2_inverse
MODEL=stage123v2_monoori_woSIG_srcsample2_roi_maskloss_depth
#MODEL=stage123v2_monoori_woSIG_srcsample2_roi_maskloss_depth
mkdir ${OUTDIR}/${MODEL}
for EPOCH in {8..8..-1}
    do
    OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
    echo ${OUTPATH}
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth.py \
        --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
        --batch_size 8 --num_workers 4 --eval_mono --png > ${OUTPATH}
        #--batch_size 8 --num_workers 8 --eval_mono --png --SIG --height 320 --width 1024 > ${OUTPATH}
    done

python3 result_eval_depth/print_value.py result_eval_depth/${MODEL}
