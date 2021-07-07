#!/bin/bash
BASEDIR=../MonoDepth2/models
CUDAIDX=7
FLOWDIR=output/flow_raw
OUTDIR=result_eval_flow_raw
MODEL=mono_640x192
EPOCH=0

mkdir ${OUTDIR}/${MODEL}
# step1: compute flow
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow3.py \
    --load_weights_folder ${BASEDIR}/${MODEL}/ \
    --input_dir dataset/data_scene_flow/training/image_2 \
    --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
    --batch_size 1 \
    --dataset 'kitti_raw_data_flow'
    
# step2: npy2png
CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 flow_npy2png.py ${FLOWDIR}/${MODEL}/weights_${EPOCH}

# step3: eval flow in 3 types
for TYPE in all fg bg
do
    mkdir ${OUTDIR}/${MODEL}/${TYPE}
    CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 eval_fg_flow_kitti_raw.py \
	--dataset_dir="dataset/kitti_raw_data_flow" \
	--pred_dir=${FLOWDIR}/${MODEL}/weights_${EPOCH} \
	--ins_dir="dataset/kitti_raw_data_flow/ins" \
	--flow_type=${TYPE} > ${OUTDIR}/${MODEL}/${TYPE}/weights_${EPOCH}.log
done

python3 ${OUTDIR}/print_value.py ${OUTDIR}/${MODEL}
