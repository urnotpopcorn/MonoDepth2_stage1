import sys
sys.path.insert(0, "flow_tool")
import flowlib as fl

import copy
import os
import cv2
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

base_dir = "/home/qhhuang/monodepth-project/dataset/flow_results/"
gt_flow_dir = "/home/qhhuang/monodepth-project/dataset/data_scene_flow/training/flow_occ"
# save_dir = os.path.join(base_dir, "flow_epe_map")
save_dir = "output/flow_epe_map"

# mono_pred_flow_dir = os.path.join(base_dir, "mono2_rigid_flow_mono2_depth/flow")
# our_pred_flow_dir = os.path.join(base_dir,"stage123v2_rpose_rinspose_insloss_tgtmask_fg1_recept20_maskloss1_tgt_3dp_y0_ord_ins_20201112/flow")
mono_pred_flow_dir = "output/flow/mono_640x192/png"
our_pred_flow_dir = "output/flow/stage123v2_monoori_woSIG/weights_15/png"

def draw_flow_epe_map(tu, tv, u, v, mask):
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]
    smask = mask[:]

    ind_valid = (smask != 0)

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)

    return epe

valid_range = [20, 45, 49, 95]
for i in valid_range:
    gt_flow = fl.read_kitti_png_file(os.path.join(gt_flow_dir, str(i).zfill(6)+"_10.png"))
    # our_flow = fl.read_kitti_png_file(os.path.join(our_pred_flow_dir, str(i).zfill(6)+"_10.png"))
    # mono_flow = fl.read_kitti_png_file(os.path.join(mono_pred_flow_dir, str(i).zfill(6)+"_10.png"))
    our_flow = fl.read_kitti_png_file(os.path.join(our_pred_flow_dir, str(i).zfill(6)+".png"))
    mono_flow = fl.read_kitti_png_file(os.path.join(mono_pred_flow_dir, str(i).zfill(6)+".png"))
    dst_h = gt_flow.shape[0]
    dst_w = gt_flow.shape[1]
    our_flow = fl.resize_flow(our_flow, dst_w, dst_h)
    mono_flow = fl.resize_flow(mono_flow, dst_w, dst_h)

    fig = plt.figure(figsize=(6,4))

    plt.subplot(2,1,1)
    gt_flow_mono2 = copy.deepcopy(gt_flow)
    mono2_epe_map = draw_flow_epe_map(gt_flow_mono2[:, :, 0], gt_flow_mono2[:, :, 1], 
        mono_flow[:, :, 0], mono_flow[:, :, 1], gt_flow_mono2[:, :, 2])
    plt.imshow(mono2_epe_map, cmap="jet")
    plt.axis('off')

    plt.subplot(2,1,2)
    gt_flow_our = copy.deepcopy(gt_flow)
    our_epe_map = draw_flow_epe_map(gt_flow_our[:, :, 0], gt_flow_our[:, :, 1], 
        our_flow[:, :, 0], our_flow[:, :, 1], gt_flow_our[:, :, 2])
    plt.imshow(our_epe_map, cmap="jet")
    plt.axis('off')

    fig.subplots_adjust(right=0.9)
    l = 0.92
    b = 0.12
    w = 0.015
    h = 1 - 2*b 
    rect = [l,b,w,h] 
    cbar_ax = fig.add_axes(rect) 
    cb = plt.colorbar(cax=cbar_ax)

    plt.show()
    our_vis_save_path = os.path.join(save_dir, str(i).zfill(6)+"_10.png")
    plt.savefig(our_vis_save_path, dpi = 150)
    plt.close()
