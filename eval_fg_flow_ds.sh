#!/bin/bash
BASEDIR=log
CUDAIDX=7
MODEL=stage123v2_monoori_woSIG_srcsample2_roi_y0
FLOWDIR=output/flow_ds4
OUTDIR=result_eval_flow_ds4
EPOCH=11

mkdir ${OUTDIR}/${MODEL}

# step2: npy2png
#CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 flow_npy2png.py ${FLOWDIR}/${MODEL}/weights_${EPOCH}

# step3: eval flow in 3 types
for TYPE in all fg bg
do
    mkdir ${OUTDIR}/${MODEL}/${TYPE}
    CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 eval_fg_flow_ds.py \
	--dataset_dir="dataset/RAFT/submit_ds/flow_select_150" \
	--pred_dir=${FLOWDIR}/${MODEL}/weights_${EPOCH} \
	--ins_dir="dataset/ds_scene_flow_SIG/ins" \
	--flow_type=${TYPE} > ${OUTDIR}/${MODEL}/${TYPE}/weights_${EPOCH}.log
done
