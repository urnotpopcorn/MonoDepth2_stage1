# python3 draw_3d_motion_map.py output/flow/mono_640x192_test/weights_/npy/ output/vis_sceneflow
# SF_DIR=../TrianFlow/output/sceneflow/pseudogt_sceneflow/npy/
# MODEL=mono_640x192

MODEL=gt
SF_DIR=../TrianFlow/output/sceneflow_bgmask_noc/pseudogt_sceneflow_tgt/npy/
python3 draw_3d_motion_map.py ${SF_DIR} output/vis_sceneflow/${MODEL}

# MODEL=mono_640x192
# EPOCH=
# SF_DIR=/home/xzwu/xzwu/Code/MonoDepth2_stage1/output/sceneflow/${MODEL}/weights_${EPOCH}/npy
# python3 draw_3d_motion_map.py ${SF_DIR} output/vis_sceneflow/${MODEL}

# MODEL=geonet
# EPOCH=
# SF_DIR=/home/xzwu/xzwu/Code/MonoDepth2_stage1/output/sceneflow/${MODEL}/weights_${EPOCH}/npy
# python3 draw_3d_motion_map.py ${SF_DIR} output/vis_sceneflow/${MODEL}

# MASK_TYPE=tgt 
# MODEL=stage12_srcwarp_2_bmm
# EPOCH=18 
# SF_DIR=/home/xzwu/xzwu/Code/MonoDepth2_stage1/output/sceneflow_${MASK_TYPE}_wflow4/${MODEL}/weights_${EPOCH}/npy
# python3 draw_3d_motion_map.py ${SF_DIR} output/vis_sceneflow_${MASK_TYPE}_wflow4/${MODEL}

# MODEL=stage123v2_monoori_woSIG_srcsample2_roi_y0 
# EPOCH=11
# SF_DIR=/home/xzwu/xzwu/Code/MonoDepth2_stage1/output/sceneflow_${MASK_TYPE}_wflow4/${MODEL}/weights_${EPOCH}/npy
# mkdir -p output/vis_sceneflow_${MASK_TYPE}_wflow4/${MODEL} 
# python3 draw_3d_motion_map.py ${SF_DIR} output/vis_sceneflow_${MASK_TYPE}_wflow4/${MODEL} 

# MODEL=from_scratch_delta
# EPOCH=13
# SF_DIR=/home/xzwu/xzwu/Code/MonoDepth2_deformation/output/sceneflow/${MODEL}/weights_${EPOCH}/npy
# python3 draw_3d_motion_map.py ${SF_DIR} output/vis_sceneflow/${MODEL}

# MODEL=ori_pretrained3
# SF_DIR=../LiHanhan/output/sceneflow/ori_pretrained3
# mkdir output/vis_sceneflow/${MODEL}
# python3 draw_3d_motion_map.py ${SF_DIR} output/vis_sceneflow/${MODEL}
