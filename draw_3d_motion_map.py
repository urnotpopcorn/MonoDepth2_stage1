#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import os
import PIL
import math
import PIL.Image
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import warnings
import time
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './flow_tool/')
import flowlib as fl
#from torchvision import transforms

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

def compute_flow_mask(gt_sceneflow):
    # flow mask
    c, h, w = gt_sceneflow.shape
    mask_flow = gt_sceneflow[3]
    mask_flow = rescale_mask(mask_flow, h, w)

    return mask_flow

def rescale_sceneflow(gt_flow, pred_flow, mask=None):
    if mask is not None:
        # valid_gt_flow = gt_flow[0, :, :][mask]
        # valid_pred_flow = pred_flow[0, :, :][mask]
        # # mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        # # valid_gt_flow = gt_flow[mask]
        # # valid_pred_flow = pred_flow[mask]
        # ratio = np.median(valid_gt_flow) / np.median(valid_pred_flow)

        mask = np.expand_dims(mask, 0).repeat(3, axis=0)
        valid_gt_flow = (gt_flow * mask)
        valid_pred_flow = (pred_flow * mask)

        # print(np.median(valid_gt_flow[valid_gt_flow>1e-3]), np.median(valid_pred_flow[valid_pred_flow>1e-3]))
        if np.sum(valid_pred_flow >1e-3) < 1e-3:
            ratio = 0
        else:
            ratio = np.median(valid_gt_flow[valid_gt_flow>1e-3]) / (np.median(valid_pred_flow[valid_pred_flow>1e-3])+1e-3) 
    else:
        ratio = np.median(gt_flow) / np.median(pred_flow)
    
    pred_flow *= ratio

    return pred_flow

def compute_mask(gt_depth):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    gt_height, gt_width = gt_depth.shape[:2]
    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                        0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)

    return mask
    
def rescale_mask(mask, h, w, thres=0.1):
    mask = np.float32(mask)
    mask = cv2.resize(mask * 255.0, (w, h)) / 255.0
    
    mask[mask >= thres] = 1
    mask[mask < thres] = 0
    mask = mask.astype(bool)
    
    return mask

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def compute_valid_mask(gt_depth):
    gt_height, gt_width = gt_depth.shape[:2]
    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                        0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    
    return mask

def compute_mask_based_gtdepth(i, height, width, gt_flow):
    depth_dir = '/home/xzwu/xzwu/Code/TrianFlow/output/depth/gt_sceneflow'
    input_file = str(i).zfill(6)
    if not os.path.exists(os.path.join(depth_dir, input_file+'_10.npy')):
        mask = np.ones((height, width))
        return mask
        
    depth1 = np.load(os.path.join(depth_dir, input_file+'_10.npy'))
    depth2 = np.load(os.path.join(depth_dir, input_file+'_11.npy'))
    mask1 = compute_valid_mask(depth1)
    mask2 = gt_flow[4]
    mask1 = rescale_mask(mask1, height, width)
    mask2 = rescale_mask(mask2, height, width)
    
    mask = np.logical_and(mask1, mask2)
    mask = rescale_mask(mask, height, width)

    return mask

def get_sem_ins(sem_ins_path):
    sem_ins = np.load(sem_ins_path)
    return sem_ins

def convert_ins_onehot(input_var):
    max_num = int(np.max(input_var))
    ins_tensor = torch.Tensor(np.asarray(input_var))
    K_num = 5
    if max_num+1 <= K_num:
        ins_tensor_one_hot = torch.nn.functional.one_hot(ins_tensor.to(torch.int64), K_num).type(torch.bool)
    else:
        ins_tensor_one_hot = torch.nn.functional.one_hot(ins_tensor.to(torch.int64), max_num+1).type(torch.bool)

    ins_tensor_one_hot = ins_tensor_one_hot[:,:,:K_num]

    # return ins_tensor_one_hot.permute(2,0,1).unsqueeze(0).numpy()
    return ins_tensor_one_hot.permute(2,0,1).numpy()
    
def extract_bbox_ins_edge(ins_dir, idx, height, width):
    # ins_path = os.path.join(ins_dir, file_name)
    ins_path = os.path.join(ins_dir, '%.6d_10.npy' % idx)

    ins_seg = get_sem_ins(ins_path)
    sig_ins_id_seg = PIL.Image.fromarray(np.uint8(ins_seg[:,:,1])).convert("L")
    ins_width, ins_height = sig_ins_id_seg.size

    sig_ins_id_seg = sig_ins_id_seg.resize((width, height), PIL.Image.NEAREST)
    ins_onehot = convert_ins_onehot(sig_ins_id_seg)

    return ins_onehot
    
def load_fg_mask(ins_dir, idx, height, width):
    valid_class = [1,2,3,4,6,7,8,16,17,18,19,20,21,22,23,24]
    dynamic_fg_ins = np.load(os.path.join(ins_dir, '%.6d_10.npy' % idx))[:,:,0]
    objs = [dynamic_fg_ins==i for i in valid_class]
    fg_mask = np.expand_dims(np.sum(objs, axis=0),2) * 1.0
    # fg_mask = cv2.resize(np.array(255.0*fg_mask), (width, height))
    fg_mask = rescale_mask(fg_mask, height, width)

    return fg_mask

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

# def zerozation(data):
#     np.min()
#     mu = np.mean(data)
#     sigma = np.std(data)
#     return (data - mu) / sigma

def preprocess_sceneflow(gt_flow, gt_depth, scene_flow):
    mask = compute_mask(gt_depth)
    _, h, w = gt_flow.shape
    mask = rescale_mask(mask, h, w)
    scene_flow = rescale_sceneflow(gt_flow, scene_flow, mask)
    return scene_flow

# def draw_sceneflow_per_image_RGB(scene_flow, vis_save_path, 
#                                     src_img=None, tgt_img=None, mask=None,
#                                     gt_flow=None, gt_depth=None):
#     if mask is not None and np.max(mask) > 1.0:
#         mask = mask / 255.0

#     # preprocess
#     # scene_flow = 1.0 / (scene_flow+1e-3)
#     scene_flow[scene_flow<0] = -np.log(-scene_flow[scene_flow<0])
#     scene_flow[scene_flow>0] = np.log(scene_flow[scene_flow>0])
#     # scene_flow = preprocess_sceneflow(gt_flow, gt_depth, scene_flow)
#     # scene_flow = scene_flow / np.max(np.abs(scene_flow))
#     print(np.mean(scene_flow), np.max(scene_flow), np.min(scene_flow))

#     # normal
#     scene_flow = normalization(scene_flow)

#     scene_flow = np.transpose(scene_flow, (1,2,0)) 
#     mask = np.expand_dims(mask, -1)

#     plt.figure()
#     plt.imshow(scene_flow * mask)
#     plt.axis('off')
#     plt.savefig(vis_save_path.replace('.jpg', '_sceneflow_mask.jpg'), bbox_inches='tight', pad_inches=0.0)
#     plt.close()

#     plt.figure()
#     plt.imshow(scene_flow)
#     plt.axis('off')
#     plt.savefig(vis_save_path.replace('.jpg', '_sceneflow.jpg'), bbox_inches='tight', pad_inches=0.0)
#     plt.close()

def draw_sceneflow_sceneflow(scene_flow, vis_save_path, src_img=None, tgt_img=None, mask=None):
    if mask is not None and np.max(mask) > 1.0:
        mask = mask / 255.0
        mask = mask.astype(float)

    # mask = scene_flow[3].astype(float)
    # scene_flow = (scene_flow + 127.0 ) / 255.0 
    # sceneflow # 2, 128, 416, 3
    
    # LiHanhan:
    '''
    scene_flow = np.transpose(scene_flow[0], (2,0,1))
    if mask is not None:
        scene_flow[0] = scene_flow[0] * mask
        scene_flow[1] = scene_flow[1] * mask
        scene_flow[2] = scene_flow[2] * mask
    scene_flow = scene_flow * 5 + 0.5
    '''
    scene_flow[0] = scene_flow[0] * mask
    scene_flow[1] = scene_flow[1] * mask
    scene_flow[2] = scene_flow[2] * mask
    
    # scene_flow = scene_flow * 10 + 0.5
    # scene_flow = scene_flow * 0.2 + 0.5 #geonet
    # scene_flow = scene_flow * 5 + 0.5 #gt
    scene_flow = scene_flow + 0.5

    x = scene_flow[0]
    y = scene_flow[1]
    z = scene_flow[2]

    scene_flow_new = np.stack([x, y, z], -1)
    abs_max = 0.1

    plt.figure()
    plt.imshow(scene_flow_new)
    # print(np.mean(scene_flow_new), np.min(scene_flow_new), np.max(scene_flow_new), np.mean(mask))
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_scene_flow.jpg'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

def draw_sceneflow_XYZ(scene_flow, vis_save_path, src_img=None, tgt_img=None, mask=None):
    if mask is not None and np.max(mask) > 1.0:
        mask = mask / 255.0
    elif mask is None:
        mask = scene_flow[3].astype(float)

    # scene_flow = scene_flow * 5
    # abs_max = 1
    abs_max = 1
    # abs_max = 10 #gt

    x = scene_flow[0] 
    y = scene_flow[1]
    z = scene_flow[2]

    plt.figure()
    x = x * mask #gt
    plt.imshow(x , cmap = "RdBu", vmin=-abs_max, vmax=abs_max)
    # print(np.mean(x[x!=0]), np.min(x[x!=0]), np.max(x[x!=0]))
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_x.jpg'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    y = y * mask #gt
    plt.imshow(y, cmap = "PiYG", vmin=-abs_max, vmax=abs_max)
    # print(np.mean(y[y!=0]), np.min(y[y!=0]), np.max(y[y!=0]))
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_y.jpg'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    z = z * mask #gt
    plt.imshow(z, cmap = "RdBu", vmin=-abs_max, vmax=abs_max)
    # print(np.mean(z[z!=0]), np.min(z[z!=0]), np.max(z[z!=0]))
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_z.jpg'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.imshow(mask)
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_mask.jpg'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

# def draw_sceneflow(scene_flow, vis_save_path, 
#                     src_img=None, tgt_img=None, mask=None,
#                     gt_flow=None, gt_depth=None):
    
#     if mask is not None and np.max(mask) > 1.0:
#         mask = mask / 255.0

#     x = scene_flow[0]
#     y = scene_flow[1]
#     z = scene_flow[2]

#     plt.figure(figsize=(12,20))

#     abs_max = 0.1

#     plt.subplot(6,1,1)
#     plt.imshow(src_img)
#     plt.title("source (t+1)")

#     plt.subplot(6,1,2)
#     plt.imshow(tgt_img)
#     plt.title("target (t)")

#     # plt.subplot(6,1,3)
#     # # plt.imshow(x, cmap = "Reds", vmin=0, vmax=1)
#     # plt.imshow(x / np.max(np.abs(scene_flow)), cmap = "bwr", vmin=-abs_max, vmax=abs_max)
#     # plt.title("delta x")
#     # plt.colorbar()


#     # plt.subplot(6,1,4)
#     # # plt.imshow(y, cmap = "Greens", vmin=0, vmax=1)
#     # plt.imshow(y / np.max(np.abs(scene_flow)), cmap = "bwr", vmin=-abs_max, vmax=abs_max)
#     # plt.title("delta y")
#     # plt.colorbar()

#     # plt.subplot(6,1,5)
#     # # plt.imshow(-z, cmap = "Blues", vmin=0, vmax=1)
#     # plt.imshow(z / np.max(np.abs(scene_flow)), cmap = "bwr", vmin=-abs_max, vmax=abs_max)
#     # plt.title("delta z")
#     # plt.colorbar()

#     scene_flow = np.transpose(scene_flow, (1,2,0)) 
#     mask = np.expand_dims(mask, -1)

#     # direction
#     moving_direction = np.ones_like(scene_flow) * 0.5 # 0.5 represents no motion
#     moving_direction[scene_flow < 0] = 0.0
#     moving_direction[scene_flow > 0] = 1.0
#     # moving_direction = np.exp(moving_direction)

#     # rgb 
#     # scene_flow[scene_flow < 0] = -np.log(-scene_flow[scene_flow < 0])
#     # scene_flow[scene_flow > 0] = np.log(scene_flow[scene_flow > 0])
#     # print(np.mean(scene_flow), np.max(scene_flow), np.min(scene_flow))
#     scene_flow = np.abs(scene_flow) * 10
#     # scene_flow = normalization(scene_flow) # [e^-1, e] -> [0, 1]
#     # print(np.mean(scene_flow), np.max(scene_flow), np.min(scene_flow))

#     plt.subplot(6,1,3)
#     plt.imshow(scene_flow)
#     plt.title("sceneflow")
#     # plt.colorbar()

#     plt.subplot(6,1,4)
#     plt.imshow(scene_flow)
#     plt.title("abs")
#     # plt.colorbar()

#     plt.subplot(6,1,5)
#     plt.imshow(moving_direction)
#     plt.title("direction")
    
#     # if mask is not None:
#     #     plt.subplot(6,1,5)
#     #     plt.imshow(mask)
#     #     plt.title("mask")

#     # plt.show()
#     make_dir(os.path.dirname(vis_save_path))
#     plt.savefig(vis_save_path)
#     plt.close()

def load_img_pair(input_dir, input_file, img_ext="png"):
    input_path_1 = os.path.join(input_dir, input_file+"_10."+img_ext)
    input_path_2 = os.path.join(input_dir, input_file+"_11."+img_ext)

    #[h, w, 3] -> [3, h, w]->[1, 3, h, w]
    '''
    resize_func = transforms.Resize((self.opt.height, self.opt.width),
                                    interpolation=PIL.Image.ANTIALIAS)

    img1 = resize_func(PIL.Image.open(input_path_1).convert('RGB'))
    img2 = resize_func(PIL.Image.open(input_path_2).convert('RGB'))
    '''
    img1 = PIL.Image.open(input_path_1).convert('RGB')
    img2 = PIL.Image.open(input_path_2).convert('RGB')

    return img1, img2

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    src_dir = "/home/xzwu/xzwu/Code/MonoDepth2_stage1/dataset/data_scene_flow/training/image_2"
    ins_dir = "/home/xzwu/xzwu/Code/MonoDepth2_stage1/dataset/data_scene_flow_SIG/ins"
    # gt_depth_dir = "../MonoDepth2_splits/eigen/"
    # gt_depth_path = os.path.join(gt_depth_dir, "gt_depths.npz")
    # gt_depths = np.load(gt_depth_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]
    gt_flow_dir = "/home/xzwu/xzwu/Code/TrianFlow/output/sceneflow_bgmask_noc/pseudogt_sceneflow_tgt/npy/"

    # selected_idx = [11, 12, 15]
    selected_idx = [18,19,20, 45, 46, 49, 58, 91, 95, 100, 184, 193, 197] # 43, 44, 
    # selected_idx = [2,20,46] # 43, 44, 
    # for idx in tqdm(range(200)):
    for idx in tqdm(selected_idx):
        input_file = '%.6d.npy' % idx
        input_path = os.path.join(input_dir, input_file)
        if not os.path.exists(input_path):
            continue
        
        scene_flow = np.load(input_path)
        tgt_img, src_img = load_img_pair(src_dir, input_file.split('.')[0])

        gt_flow_path = os.path.join(gt_flow_dir, input_file)
        if not os.path.exists(gt_flow_path):
            print(gt_flow_path)
            continue
        gt_flow = np.load(gt_flow_path)
        # gt_depth = gt_depths[idx]

        _, h, w = scene_flow.shape
        flow_mask = compute_flow_mask(gt_flow)
        depth_mask = compute_mask_based_gtdepth(idx, h, w, gt_flow)
        fg_mask = load_fg_mask(ins_dir, idx, 192, 640)
        mask = flow_mask #np.logical_and(flow_mask, depth_mask)
        # print(np.sum(mask))

        # mask = extract_bbox_ins_edge(ins_dir, idx, h, w)[2]
        # mask = load_fg_mask(ins_dir, idx, 128, 416) # GeoNet
        # mask = load_fg_mask(ins_dir, idx, 192, 640)

        output_path = os.path.join(output_dir, input_file.replace(".npy", ".jpg"))
        
        draw_sceneflow_sceneflow(scene_flow, output_path, 
                                src_img, tgt_img, mask)
        draw_sceneflow_XYZ(scene_flow, output_path, 
                                src_img, tgt_img, mask)

if __name__ == '__main__':
    main()
