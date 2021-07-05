# /local/xjqi/monodepth-project/monodepth2/splits/eigen_zhou/train_files.txt
# 2011_09_26/2011_09_26_drive_0028_sync 421 r

# /local/xjqi/monodepth-project/monodepth2/splits/eigen_zhou/val_files.txt
# 2011_09_26/2011_09_26_drive_0104_sync 80 r

# /local/xjqi/monodepth-project/monodepth2/splits/eigen/test_files.txt
# 2011_10_03/2011_10_03_drive_0047_sync 0000000768 l

import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger

import os
import numpy as np
import cv2
from PIL import Image

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from tqdm import tqdm

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def inference(predictor, kitti_data_dir, kitti_data_ins_dir, img_files, mode):
    # train_files = "/local/xjqi/monodepth-project/monodepth2/splits/eigen_zhou/train_files.txt"
    # # 2011_09_26/2011_09_26_drive_0028_sync 421 r

    # val_files = "/local/xjqi/monodepth-project/monodepth2/splits/eigen_zhou/val_files.txt"
    # # 2011_09_26/2011_09_26_drive_0104_sync 80 r

    # test_files = "/local/xjqi/monodepth-project/monodepth2/splits/eigen/test_files.txt"
    # # 2011_10_03/2011_10_03_drive_0047_sync 0000000768 l

    # /local/xjqi/monodepth-project/kitti_data/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000768.png

    img_path_files = []
    for img_path in img_files:
        img_index = img_path.split(" ")[1].zfill(10) # 0000000421
        if img_path.split(" ")[2] == "l":
            img_dir = "02"
        elif img_path.split(" ")[2] == "r":
            img_dir = "03"

        img_path_f = os.path.join(img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index)+".png")
        img_path_files.append(img_path_f)

    for img_path in tqdm(img_path_files):
        # img_path: 2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000768.png
        # src_img_path: /local/xjqi/monodepth-project/kitti_data/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000768.png
        
        src_img_path = os.path.join(kitti_data_dir, img_path)
        rgb_img = np.array(Image.open(src_img_path).convert('RGB'))
        # crop and convert RGB to BGR
        rgb_img = rgb_img[:, :, ::-1].copy() 

        output = predictor(rgb_img)

        mask = output['instances'].pred_masks.cpu().numpy().transpose([1,2,0])

        ins_class = output['instances'].pred_classes.cpu().numpy()
        ins_1_0 = np.zeros((mask.shape[0],mask.shape[1]), dtype=np.int8)
        ins_1_1 = np.zeros((mask.shape[0],mask.shape[1]), dtype=np.int8)

        for i, sig_class in enumerate(ins_class):
            ins_1_0[mask[:,:,i]] = sig_class+1

        for i, sig_class in enumerate(ins_class):
            ins_1_1[mask[:,:,i]] = i+1
            
        ins_pack_0 = np.expand_dims(ins_1_0, axis=2)
        ins_pack_1 = np.expand_dims(ins_1_1, axis=2) 

        ins_cat = np.concatenate((ins_pack_0, ins_pack_1), axis=2)

        dst_img_path = img_path.replace("png", "npy")
        save_img_path = os.path.join(kitti_data_ins_dir, mode, dst_img_path)
        make_dir(os.path.dirname(save_img_path))

        np.save(save_img_path, ins_cat)


def get_extra_lines(kitti_data_dir, kitti_data_ins_sem_dir, img_files, mode):
    # train_lines = get_extra_lines(kitti_data_dir, kitti_data_sem_dir, test_lines, mode="train")
    # 2011_09_30/2011_09_30_drive_0028_sync 2300 r

    img_path_files = []
    for img_path in img_files:
        frame_instant = int(img_path.split(" ")[1])
        t_frame_0 = str(frame_instant - 1)
        t_frame_1 = str(frame_instant)
        t_frame_2 = str(frame_instant + 1)

        img_index_0 = t_frame_0.zfill(10)
        img_index_1 = t_frame_1.zfill(10)
        img_index_2 = t_frame_2.zfill(10)

        if img_path.split(" ")[2] == "l":
            img_dir = "02"
        elif img_path.split(" ")[2] == "r":
            img_dir = "03"

        t_frame_0_src = os.path.join(kitti_data_ins_sem_dir, mode, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_0)+".npy")
        t_frame_1_src = os.path.join(kitti_data_ins_sem_dir, mode, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_1)+".npy")
        t_frame_2_src = os.path.join(kitti_data_ins_sem_dir, mode, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_2)+".npy")

        t_frame_0_dst = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_0)+".png")
        t_frame_1_dst = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_1)+".png")
        t_frame_2_dst = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_2)+".png")

        if not os.path.exists(t_frame_0_src):
            if os.path.exists(t_frame_0_dst):
                img_path_files.append(img_path.split(" ")[0]+" "+str(img_index_0)+" "+img_path.split(" ")[2])

        if not os.path.exists(t_frame_1_src):
            if os.path.exists(t_frame_1_dst):
                img_path_files.append(img_path.split(" ")[0]+" "+str(img_index_1)+" "+img_path.split(" ")[2])

        if not os.path.exists(t_frame_2_src):
            if os.path.exists(t_frame_2_dst):
                img_path_files.append(img_path.split(" ")[0]+" "+str(img_index_2)+" "+img_path.split(" ")[2])

    return img_path_files


if __name__ == "__main__":
    machine_code = "/home/qhhuang"

    train_files = os.path.join(machine_code, "monodepth-project/monodepth2/splits/geonet_flow/train_files.txt")
    # 2011_09_26/2011_09_26_drive_0028_sync 421 r
    with open(train_files, "r") as fd:
        train_lines = fd.read().splitlines()

    val_files = os.path.join(machine_code, "monodepth-project/monodepth2/splits/geonet_flow/val_files.txt")
    # 2011_09_26/2011_09_26_drive_0104_sync 80 r

    with open(val_files, "r") as fd:
        val_lines = fd.read().splitlines()

    # /userhome/34/h3567721/dataset/kitti/kitti_raw_eigen/2011_09_26_drive_0001_sync_02/***.jpg
    base_dir = './' #os.path.join(machine_code, "monodepth-project")
    kitti_data_dir = os.path.join(base_dir, "dataset", "raw_data")
    kitti_data_ins_dir = os.path.join(base_dir, "dataset", "kitti_data_ins_geonet_flow")
    make_dir(kitti_data_ins_dir)

    extra_train_lines = get_extra_lines(kitti_data_dir, kitti_data_ins_dir, train_lines, mode="train")
    extra_val_lines = get_extra_lines(kitti_data_dir, kitti_data_ins_dir, val_lines, mode="val")

    extra_train_lines = list(set(extra_train_lines))
    extra_val_lines = list(set(extra_val_lines))

    print(len(extra_train_lines))
    print(len(extra_val_lines))

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    print("inference val")
    inference(predictor, kitti_data_dir, kitti_data_ins_dir, extra_val_lines, mode="val")
    print("inference train")
    inference(predictor, kitti_data_dir, kitti_data_ins_dir, extra_train_lines, mode="train")