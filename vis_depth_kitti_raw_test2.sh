MODEL=stage1_monoori_woSIG_vkitti_wweather_max40
EPOCH=22
PRED=log/${MODEL}/models/weights_${EPOCH}/disps_eigen_split.npy
SAVE=output/vis_depth/${MODEL}/weights_${EPOCH}
python3 vis_depth_kitti_raw_test.py ${PRED} ${SAVE}

