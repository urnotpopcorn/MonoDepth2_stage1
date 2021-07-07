#!/bin/bash
BASEDIR=log
CUDA_IDX=3
# MODEL=stage123v2_monoori_woSIG_depth
#MODEL_DIR=../MonoDepth2/models/mono_640x192
MODEL=stage1_monoori_woSIG_vkitti_woweather_max40

for EPOCH in {17..17..-1}
    do
    # OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
    # echo ${OUTPATH}
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 save_disp_kitti_raw_test.py \
        --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
        --batch_size 8 --num_workers 8 --eval_mono --png 
        # --SIG --height 320 --width 1024 > ${OUTPATH}
        # --load_weights_folder ${MODEL_DIR}/ \
        # --batch_size 8 --num_workers 4 --eval_mono --png #> ${OUTPATH}
    done
