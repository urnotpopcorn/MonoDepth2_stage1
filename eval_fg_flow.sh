#!/bin/bash

# BASEMODEL=output/flow
# OUTDIR=result_eval_flow

# MODEL=stage12_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_roi4e-1_new
# mkdir ${OUTDIR}/${MODEL}

# for EPOCH in {16..16..-1}
#     do
#         PREDDIR=${BASEMODEL}/${MODEL}/weights_${EPOCH}
#         python3 flow_npy2png.py ${PREDDIR}
#         for TYPE in all fg bg
#             do
#                 mkdir ${OUTDIR}/${MODEL}/${TYPE}
#                 python3 eval_fg_flow.py \
#                     --dataset_dir="dataset/data_scene_flow" \
#                     --pred_dir=${PREDDIR} \
#                     --ins_dir="dataset/data_scene_flow_SIG/ins" \
#                     --flow_type=${TYPE} > ${OUTDIR}/${MODEL}/${TYPE}/weights_${EPOCH}.log
#             done
#     done


# PREDDIR='/home/xzwu/xzwu/Code/PackNet-Sfm/output/PackNet01_MR_selfsup_K/flow'
# OUTDIR='/home/xzwu/xzwu/Code/PackNet-Sfm/output/PackNet01_MR_selfsup_K/eval_flow'
# python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         mkdir ${OUTDIR}/${TYPE}
#         python3 eval_fg_flow.py \
#             --dataset_dir="dataset/data_scene_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/data_scene_flow_SIG/ins" \
#             --flow_type=${TYPE} > ${OUTDIR}/${TYPE}/weights_0.log
#     done


# PREDDIR='/home/xzwu/xzwu/Code/MonoDepth2_stage1/output/flow/mono_640x192'
# OUTDIR='/home/xzwu/xzwu/Code/MonoDepth2_stage1/result_eval_flow/mono_640x192'
# PREDDIR='/home/xzwu/xzwu/Code/MonoDepth2_stage1/output/flow/mono_odom_640x192'
# OUTDIR='/home/xzwu/xzwu/Code/MonoDepth2_stage1/result_eval_flow/mono_odom_640x192'
# PREDDIR='/home/xzwu/xzwu/Code/MonoDepth2_stage1/output/flow/mono_640x192_qh'
# OUTDIR='/home/xzwu/xzwu/Code/MonoDepth2_stage1/result_eval_flow/mono_640x192_qh'

OUTDIR=result_eval_flow
FLOWDIR=output/flow
MODEL=geonet_rigid_flow
EPOCH=0
mkdir ${OUTDIR}/${MODEL}

for TYPE in all fg bg
    do
	PREDDIR=${FLOWDIR}/${MODEL}/weights_${EPOCH}
        echo ${PREDDIR}
	mkdir ${OUTDIR}/${MODEL}/${TYPE}
	python3 eval_fg_flow.py \
            --dataset_dir="dataset/data_scene_flow" \
            --pred_dir=${PREDDIR} \
            --ins_dir="dataset/data_scene_flow_SIG/ins" \
            --flow_type=${TYPE} > ${OUTDIR}/${MODEL}/${TYPE}/weights_${EPOCH}.log
    done
