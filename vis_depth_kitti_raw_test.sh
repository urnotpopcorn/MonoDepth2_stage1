MODEL=stage1_monoori_woSIG_vkitti_woweather_max40
EPOCH=17
PRED=log/${MODEL}/models/weights_${EPOCH}/disps_eigen_split.npy
SAVE=output/vis_depth/${MODEL}/weights_${EPOCH}
CUDA_IDX=3
CUDA_VISIBLE_DEVICES=${CUDA_IDX} python3 vis_depth_kitti_raw_test.py ${PRED} ${SAVE}

