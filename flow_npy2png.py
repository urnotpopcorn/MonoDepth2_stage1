from __future__ import division
import cv2
import os
import numpy as np
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path + '/flow_tool/')
import flowlib as fl
from tqdm import tqdm


base_dir = sys.argv[1]
# /mnt/sdb/xzwu/Code/MonoDepth2_eval/models/models_138/finetune_d_i_wpose_winspose_newenv_areasize_noaug_inssize_fg25e-2_recept/weights_1"

npy_dir = os.path.join(base_dir, "npy")

png_dir = os.path.join(base_dir, "png")
if not os.path.exists(png_dir):
    os.makedirs(png_dir)

color_dir  = os.path.join(base_dir, 'color')

if not os.path.exists(color_dir):
    os.makedirs(color_dir)

# for img in tqdm(os.listdir(npy_dir)):
#     pred_flow = np.load(os.path.join(npy_dir, img), allow_pickle=True)
#     png_path = os.path.join(png_dir, img.replace("npy", "png"))
#     mask_blob   = np.ones((192, 640), dtype = np.uint16)
#     fl.write_kitti_png_file(png_path, pred_flow, mask_blob)

#     color_path    = os.path.join(color_dir, img.replace("npy", "png"))
#     color_flow  = fl.flow_to_image(pred_flow)
#     color_flow  = cv2.cvtColor(color_flow, cv2.COLOR_RGB2BGR)
#     color_flow  = cv2.imwrite(color_path, color_flow)
file_cnt = len(os.listdir(npy_dir))
for i in tqdm(range(file_cnt)):
    try:
        img_name = str(i).zfill(6) + ".npy"
        pred_flow = np.load(os.path.join(npy_dir, img_name), allow_pickle=True)

        png_path = os.path.join(png_dir, str(i).zfill(6) + ".png")
        mask_blob   = np.ones((pred_flow.shape[0], pred_flow.shape[1]), dtype = np.uint16)
        fl.write_kitti_png_file(png_path, pred_flow, mask_blob)

        color_path    = os.path.join(color_dir, str(i).zfill(6) + ".png")
        color_flow  = fl.flow_to_image(pred_flow)
        color_flow  = cv2.cvtColor(color_flow, cv2.COLOR_RGB2BGR)
        color_flow  = cv2.imwrite(color_path, color_flow)
    except Exception as e:
        print(e)
