LOGDIR=log
MODEL=stage1_woPool_odo
IDX=10

for EPOCH in {19..0..-1}
do
	PRE=${LOGDIR}/${MODEL}/models/weights_${EPOCH}
	#PRE=../MonoDepth2/models/mono_640x192

	CUDA_VISIBLE_DEVICES=7 python3 evaluate_pose.py --eval_split odom_${IDX} \
		--load_weights_folder ${PRE} \
		--data_path dataset/Kitti/kitti_odometry
done
