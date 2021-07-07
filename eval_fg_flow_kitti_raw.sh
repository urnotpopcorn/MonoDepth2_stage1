OUTDIR=result_eval_flow_raw
FLOWDIR=output/flow_raw
MODEL=geonet_res_flow
mkdir ${OUTDIR}/${MODEL}

for EPOCH in {0..0..-1}
do
    PREDDIR=${FLOWDIR}/${MODEL}/weights_${EPOCH}
    for TYPE in all fg bg
        do
            mkdir ${OUTDIR}/${MODEL}/${TYPE}
            python3 eval_fg_flow_kitti_raw.py \
                --dataset_dir="dataset/kitti_raw_data_flow" \
                --pred_dir=${PREDDIR} \
                --ins_dir="dataset/kitti_raw_data_flow/ins" \
                --flow_type=${TYPE} > ${OUTDIR}/${MODEL}/${TYPE}/weights_${EPOCH}.log
        done
done

python3 ${OUTDIR}/print_value.py ${OUTDIR}/${MODEL}
