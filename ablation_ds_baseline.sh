CUDA_VISIBLE_DEVICES=5  python ../train_ds.py --model_name M_640x288_ds_eigen_1120 \
    --data_path /home/qhhuang/monodepth-project/dataset/drivingstereo \
    --log_dir /home/qhhuang/monodepth-project/tmp \
    --batch_size 8 \
    --num_workers 8 \
    --height 288 \
    --width 640 \
    --dataset drivingstereo_eigen \
    --split drivingstereo_eigen

CUDA_VISIBLE_DEVICES=7  python ../train_ds.py --model_name M_640x288_ds_eigen_SIG_depth_pose_1119 \
    --data_path /home/qhhuang/monodepth-project/dataset/drivingstereo \
    --log_dir /home/qhhuang/monodepth-project/tmp \
    --batch_size 8 \
    --num_workers 8 \
    --height 288 \
    --width 640 \
    --SIG \
    --dataset drivingstereo_eigen \
    --split drivingstereo_eigen

CUDA_VISIBLE_DEVICES=6  python ../train_ds.py --model_name M_640x288_ds_eigen_ignore_fg_loss_SIG_depth_pose_1119 \
    --data_path /home/qhhuang/monodepth-project/dataset/drivingstereo \
    --log_dir /home/qhhuang/monodepth-project/tmp \
    --batch_size 8 \
    --num_workers 8 \
    --height 288 \
    --width 640 \
    --SIG \
    --SIG_ignore_fg_loss \
    --dataset drivingstereo_eigen \
    --split drivingstereo_eigen