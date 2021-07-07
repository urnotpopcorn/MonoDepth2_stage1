#!/bin/bash
BASEDIR=log
CUDAIDX=0
FLOWDIR=output/flow
OUTDIR=result_eval_flow

for MODEL in 'stage123v2_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_ins_20201112'
do
    mkdir ${OUTDIR}/${MODEL}
    for EPOCH in {0..0..-1}
    do
        # step1: compute flow
        CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 compute_flow_with_ins_flow3.py \
            --load_weights_folder ${BASEDIR}/${MODEL}/models/weights_${EPOCH} \
            --input_dir dataset/data_scene_flow/training/image_2 \
            --output_dir ${FLOWDIR}/${MODEL}/weights_${EPOCH} \
            --batch_size 1 \
            --SIG \
            --instance_pose \
            --instance_motion \
            --instance_motion_v2 \
            --disable_pose_invert \
            --disable_inspose_invert \
            --ext_recept_field \
            --eval_flow_mask_outer \
            --filter_mask \
            --eval_flow_filter_warping_error_stage2 \
            --eval_flow_filter_warping_error_stage3 \
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

python3 result_eval_flow/print_value.py result_eval_flow/${MODEL}