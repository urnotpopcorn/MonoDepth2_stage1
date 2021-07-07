#!/bin/bash
BASEDIR=log
CUDAIDX=1
FLOWDIR=output/flow4
OUTDIR=result_eval_flow4

for MODEL in 'stage12_srcwarp_2_bmm'
do
    mkdir ${OUTDIR}/${MODEL}
    for EPOCH in {19..0..-1}
    do 
        # step1: compute flow
        CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow4.py \
            --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
            --input_dir dataset/data_scene_flow/training/image_2 \
            --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
            --batch_size 1 \
            --instance_pose \
            --disable_pose_invert \
            --disable_inspose_invert \
            --ext_recept_field \
            --filter_mask \
            --eval_flow_filter_warping_error_stage2 \
            --roi_diff_thres 0.2 \
            --set_y_zero \
            --use_depth_ordering

        # step2: npy2png
        CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 flow_npy2png.py ${FLOWDIR}/${MODEL}/weights_${EPOCH}

        # step3: eval flow in 3 types
        for TYPE in all fg bg
        do
            mkdir ${OUTDIR}/${MODEL}/${TYPE}
            CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 eval_fg_flow.py \
                --dataset_dir="dataset/data_scene_flow" \
                --pred_dir=${FLOWDIR}/${MODEL}/weights_${EPOCH} \
                --ins_dir="dataset/data_scene_flow_SIG/ins" \
                --flow_type=${TYPE} > ${OUTDIR}/${MODEL}/${TYPE}/weights_${EPOCH}.log
        done
    done
done

python3 ${OUTDIR}/print_value.py ${OUTDIR}/${MODEL}
