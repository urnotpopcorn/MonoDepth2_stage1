#!/bin/bash
BASEDIR=log
OUTDIR=result_eval_depth_ds
CUDA_IDX=7
# half

# MODEL=stage12_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_roi4e-1_new_ds_20201121
# mkdir ${OUTDIR}/${MODEL}
# for EPOCH in {11..0..-1}
#     do
#     OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
#     echo ${OUTPATH}
#     CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_select_ds.py \
#         --data_path dataset/drivingstereo \
#         --eval_split drivingstereo_eigen_half \
#         --eval_mono --height 288 --width 640 \
#         --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
#         --num_workers 8 --batch_size 8 > ${OUTPATH}
#     done

# MODEL=stage1_rpose
# mkdir ${OUTDIR}/${MODEL}
# for EPOCH in {15..15..-1}
#     do
#     OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
#     echo ${OUTPATH}
#     CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_select_ds.py \
#         --data_path dataset/drivingstereo \
#         --eval_split drivingstereo_eigen_half \
#         --eval_mono \
#         --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} --SIG \
#         --num_workers 16 --batch_size 16 > ${OUTPATH}
#     done
MODEL=stage123v2_monoori_woSIG_srcsample2_roi_y0_maskloss_ds_depth
# MODEL=stage123v2_monoori_woSIG_srcsample2_roi_y0
# MODEL=mono2_ds_20201121
mkdir ${OUTDIR}/${MODEL}
for EPOCH in {19..0..-1}
    do
    OUTPATH=${OUTDIR}/${MODEL}/weights_${EPOCH}.log
    echo ${OUTPATH}
    CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_select_ds.py \
        --data_path dataset/drivingstereo \
        --eval_mono --height 288 --width 640 \
        --eval_split drivingstereo_eigen_half \
        --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
        --num_workers 16 --batch_size 16 > ${OUTPATH}
    done
python3 ${OUTDIR}/print_value.py ${OUTDIR}/${MODEL}
#--eval_mono --height 288 --width 640 \

# MODEL=../../Pretrained/MonoDepth2/mono_640x192
# CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 evaluate_depth_select_ds.py \
#     --data_path dataset/drivingstereo \
#     --eval_split drivingstereo_eigen_half \
#     --eval_mono \
#     --load_weights_folder ${MODEL} \
#     --num_workers 16 --batch_size 16

# CUDA_VISIBLE_DEVICES=4 python ../evaluate_depth_select_ds.py \
#     --data_path /home/qhhuang/monodepth-project/dataset/drivingstereo \
#     --eval_mono \
#     --eval_split drivingstereo_eigen_half \
#     --log_dir /home/qhhuang/monodepth-project/tmp \
#     --load_weights_folder M_640x288_ds_eigen_SIG_depth_pose_1119 \
#     --SIG \
#     --num_workers 16 \
#     --batch_size 16 \
#     --height 288 \
#     --width 640 \
#     --depth_test_weights 0 1 2 3 4 5 6 7 8 9 10
