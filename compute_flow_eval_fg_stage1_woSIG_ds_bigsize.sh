#!/bin/bash
BASEDIR=log
CUDAIDX=6
FLOWDIR=output/flow_ds
OUTDIR=result_eval_flow_ds
MODEL=stage1_monoori_woSIG_ds

for EPOCH in {19..8..-1}
do
	mkdir ${OUTDIR}/${MODEL}
	# step1: compute flow
	CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow3.py \
	    --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
	    --input_dir dataset/data_scene_flow/training/image_2 \
	    --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
	    --batch_size 1 --height 288 --width 640 \
	    --dataset drivingstereo_eigen 
	    
	# step2: npy2png
	CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 flow_npy2png.py ${FLOWDIR}/${MODEL}/weights_${EPOCH}

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
done

python3 ${OUTDIR}/print_value.py ${OUTDIR}/${MODEL}
