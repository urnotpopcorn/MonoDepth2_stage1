
# PREDDIR=~/xzwu/Dataset/GeoNet/raw_data_depth_test_split_res_flow
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done

# PREDDIR=~/xzwu/Dataset/GeoNet/raw_data_depth_test_split_res_flow/
# # python3 flow_npy2png.py ${PREDDIR}
# echo ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done

PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/mono_640x192
echo ${PREDDIR}
python3 flow_npy2png.py ${PREDDIR}
for TYPE in all fg bg
    do
        echo ${TYPE}
        python3 eval_fg_flow_kitti_raw.py \
            --dataset_dir="dataset/kitti_raw_data_flow" \
            --pred_dir=${PREDDIR} \
            --ins_dir="dataset/kitti_raw_data_flow/ins" \
            --flow_type=${TYPE} 
    done

# PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/stage1_rpose/weights_15
# echo ${PREDDIR}
# # python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done

# PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/stage12_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_roi4e-1_new/weights_16
# echo ${PREDDIR}
# python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done


# PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/stage123v2_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_ins_20201114/weights_11
# echo ${PREDDIR}
# python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done


# PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/stage123v2_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_ins_20201112/weights_14
# echo ${PREDDIR}
# python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done


# PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/stage12_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_roi4e-1_new/weights_16
# echo ${PREDDIR}
# python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done

# PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/stage123v2_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_ins_20201112/weights_8
# echo ${PREDDIR}
# python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done


# PREDDIR=~/xzwu/Code/MonoDepth2_stage1/output/flow_raw/stage123v2_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_ins_20201112/weights_0
# echo ${PREDDIR}
# python3 flow_npy2png.py ${PREDDIR}
# for TYPE in all fg bg
#     do
#         echo ${TYPE}
#         python3 eval_fg_flow_kitti_raw.py \
#             --dataset_dir="dataset/kitti_raw_data_flow" \
#             --pred_dir=${PREDDIR} \
#             --ins_dir="dataset/kitti_raw_data_flow/ins" \
#             --flow_type=${TYPE} 
#     done