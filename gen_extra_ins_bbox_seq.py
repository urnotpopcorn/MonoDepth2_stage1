# I/O libraries
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def inference(predictor, raw_data_dir, 
                kitti_data_ins_dir, 
                kitti_data_bbox_dir, 
                date, seq, camera):
    input_dir = os.path.join(raw_data_dir, date, seq, camera, "data")
    input_path_list = os.listdir(input_dir)
    input_len = len(input_path_list)
    for idx in tqdm(range(input_len)):
        # img_path: 2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000768.png
        # src_img_path: /local/xjqi/monodepth-project/kitti_data/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000768.png
        input_file = str(idx).zfill(10)
        src_img_path = os.path.join(input_dir, input_file+".png")
        rgb_img = np.array(Image.open(src_img_path).convert('RGB'))

        # crop and convert RGB to BGR
        rgb_img = rgb_img[:, :, ::-1].copy() 
        output = predictor(rgb_img) 

        # -------------------------ins-------------------------
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

        dst_img_path = input_file+".npy"
        save_img_path = os.path.join(kitti_data_ins_dir, date, seq, camera, "data", dst_img_path)
        make_dir(os.path.dirname(save_img_path))
        np.save(save_img_path, ins_cat)

        # -------------------------box-------------------------
        bbox = output['instances'].to("cpu").pred_boxes.tensor.numpy()
        dst_img_path = input_file+".txt"
        save_img_path2 = os.path.join(kitti_data_bbox_dir, date, seq, camera, "data", dst_img_path)
        make_dir(os.path.dirname(save_img_path2))
        np.savetxt(save_img_path2, bbox, newline="\n")


if __name__ == "__main__":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print("---------------------------------------------------------------")
    kitti_data_dir = os.path.join("dataset", "raw_data")
    kitti_data_ins_dir = os.path.join("dataset", "kitti_selected_mine", "ins")
    make_dir(kitti_data_ins_dir)
    kitti_data_bbox_dir = os.path.join("dataset", "kitti_selected_mine", "bbox")
    make_dir(kitti_data_bbox_dir)

    # pretrain_model_path ="deeplabv3_cityscapes_train/frozen_inference_graph.pb"
    # MODEL = DeepLabModel(pretrain_model_path)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # inference(predictor, kitti_data_dir, kitti_data_ins_dir)
    

    print("inference...")
    date = "2011_09_26"
    camera = "image_02"

    # seq = "2011_09_26_drive_0013_sync"
    # print(seq)
    # inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0017_sync"
    # print(seq)
    # inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0018_sync"
    # print(seq)
    # inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0022_sync"
    # print(seq)
    # inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0051_sync"
    # print(seq)
    # inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0005_sync"
    # print(seq)
    # inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, date, seq, camera)
    
    seq = "2011_09_26_drive_0113_sync"
    print(seq)
    inference(predictor, kitti_data_dir, kitti_data_ins_dir, kitti_data_bbox_dir, date, seq, camera)
    