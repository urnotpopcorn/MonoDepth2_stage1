#!/bin/bash
PREDDIR='/home/xzwu/xzwu/Code/FeatDepth/log/mono_fm_from_scratch_new'
OUTDIR='/home/xzwu/xzwu/Code/MonoDepth2_stage1/result_eval_flow/FeatDepth/mono_fm_from_scratch_new'
python3 flow_npy2png.py ${PREDDIR}
for TYPE in all fg bg
    do
        mkdir ${OUTDIR}/${TYPE}
        python3 eval_fg_flow.py \
            --dataset_dir="dataset/data_scene_flow" \
            --pred_dir=${PREDDIR} \
            --ins_dir="dataset/data_scene_flow_SIG/ins" \
            --flow_type=${TYPE} > ${OUTDIR}/${TYPE}/weights_0.log
    done
