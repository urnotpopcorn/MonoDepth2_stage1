MODEL=stage1_monoori_woSIG_vkitti_wweather_max40
EPOCH=22
PRED=output/disp_VKitti/${MODEL}/weights_${EPOCH}
MONO=output/disp_VKitti/mono_640x192/weights_0
SAVE=output/vis_depth_VKitti/${MODEL}/weights_${EPOCH}
CUDAIDX=1

CUDA_VISIBLE_DEVICES=${CUDAIDX} python3 vis_depth_vkitti.py ${PRED} ${MONO} ${SAVE}

