#!/usr/bin/env python

"""
??compute_flow_with_ins_flow3.py???,?I_s??sample,?bmm???????,?img_pred????

"""
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
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import datasets
import networks
from utils import *
from layers import *
from kitti_utils import *
from options import MonodepthOptions

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './flow_tool/')
import flowlib as fl

class Evaler:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else 3

        if self.opt.dataset == 'drivingstereo_eigen':
            K_ori = np.array([[1.14, 0,     0.518, 0],
                                [0,    2.509, 0.494, 0],
                                [0,    0,     1,     0],
                                [0,    0,     0,     1]], dtype=np.float32)
        else:                        
            K_ori = np.array([[0.58, 0,     0.5, 0],
                            [0,    1.92, 0.5, 0],
                            [0,    0,    1,   0],
                            [0,    0,    0,   1]], dtype=np.float32)

        self.K = dict()
        self.inv_K = dict()
        for scale in range(4):
            K = K_ori.copy()

            K[0, :] *= self.opt.width // (2 ** scale)
            K[1, :] *= self.opt.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            self.K[scale] = torch.from_numpy(K).float().unsqueeze(0).to(self.device)
            self.inv_K[scale] = torch.from_numpy(inv_K).float().unsqueeze(0).to(self.device)

        if self.opt.SIG:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, False, mode = "SIG", cur="depth")
        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, False)

        self.models["encoder"].to(self.device)

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        # if self.opt.SIG:
        #     self.models["pose_encoder"] = networks.ResnetEncoder(
        #         self.opt.num_layers,
        #         False,
        #         num_input_images=self.num_pose_frames*3, mode="SIG", cur="pose")
        # else:
        #     self.models["pose_encoder"] = networks.ResnetEncoder(
        #         self.opt.num_layers,
        #         False,
        #         num_input_images=self.num_pose_frames*3)
        self.models["pose_encoder"] = networks.ResnetPoseEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

        self.models["pose_encoder"].to(self.device)

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)

        self.models["pose"].to(self.device)

        if self.opt.instance_pose:
            self.models["instance_pose_encoder"] = networks.InsResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["instance_pose_encoder"].to(self.device)

            self.models["instance_pose"] = networks.InsPoseDecoder(
                num_RoI_cat_features=1024,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["instance_pose"].to(self.device)

        # --------------------------------------stage3------------------------------------------
        if self.opt.instance_motion:
            # stage3 is based on stage2
            self.models["instance_motion"] = networks.InsMotionDecoder(
                self.models["instance_pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)
            self.models["instance_motion"].to(self.device)
            
        self.load_model()

        print("Eval is using:\n  ", self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales: # [0,1,2,3]
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            # initialize backproject_depth and project_3d at each scale
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.set_eval()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def predict_poses(self, tensorImg1, tensorImg2):
        pose_inputs = [tensorImg1, tensorImg2]

        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
        # pose decoder
        axisangle, translation = self.models["pose"](pose_inputs)

        # Invert the matrix if the frame id is negative
        cam_T_cam = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=False)

        return cam_T_cam

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        models_to_load = self.opt.models_to_load
        if self.opt.instance_pose:
            models_to_load.append("instance_pose")
            models_to_load.append("instance_pose_encoder")
            
        if self.opt.instance_motion:
            models_to_load.append("instance_motion")

        for n in models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            try:
                model_dict = self.models[n].state_dict()
                pretrained_dict = torch.load(path)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.models[n].load_state_dict(model_dict)
            except Exception as e:
                print(e)

    def predict_depth(self, input_tensor):
        #step0: compute disp
        features = self.models["encoder"](input_tensor)
        outputs = self.models["depth"](features)
        disp = outputs[("disp", 0)]
        
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        #step1: compute depth
        # min_depth = 0.1, max_depth = 100
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
        
        return depth
        
    
    def predict_disp(self, input_tensor):
        #step0: compute disp
        features = self.models["encoder"](input_tensor)
        outputs = self.models["depth"](features)
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        return disp

    def predict_rigid_flow(self, cam_T_cam, depth, inv_K_dict, K_dict, img1=None):
        source_scale = 0
        inv_K = inv_K_dict[source_scale]
        K = K_dict[source_scale]
        outputs = dict()

        T = cam_T_cam # T from 0 to -1 or 0 to 1
        # cam_points of frame 0, [12, 4, 122880]
        cam_points = self.backproject_depth[source_scale](depth, inv_K)
        pix_coords = self.project_3d[source_scale](cam_points, K, T)

        if img1 is not None:
            img0_pred = F.grid_sample(img1, pix_coords)

        new_pix_coords = pix_coords.clone()

        # [-1, 1] -> [0, 1] -> [0, w], [b, h, w, 2]
        new_pix_coords = new_pix_coords / 2.0 + 0.5

        new_pix_coords[:, :, :, 0] *= (new_pix_coords.shape[2]-1) # w
        new_pix_coords[:, :, :, 1] *= (new_pix_coords.shape[1]-1) # h

        xx, yy = np.meshgrid(np.arange(0, new_pix_coords.shape[2]), np.arange(0, new_pix_coords.shape[1]))
        meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
        pix_coords = torch.from_numpy(meshgrid).unsqueeze(0).repeat(self.opt.batch_size,1,1,1).float().to(self.device) # [b,2,h,w]
        pix_coords = pix_coords.permute(0, 2, 3, 1) # [b,h,w,2]

        flow_pred = new_pix_coords - pix_coords
        
        
        if img1 is not None:
            # img
            outputs["img_stage1"] = img0_pred
        # flow
        outputs["flow_stage1"] = flow_pred

        return outputs

    def extract_bbox_from_mask(self, ins_warp_mask):
        """Compute bounding boxes from masks.
        mask: [height, width]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (x1, y1, x2, y2)].
        """
        # [b, h, w]
        mask = ins_warp_mask.squeeze(1)

        ins_warp_bbox = []
        for bs_idx in range(mask.shape[0]):
            idx_mask = mask[bs_idx, :, :].cpu().numpy() # [h, w]
            # Bounding box.
            horizontal_indicies = np.where(np.any(idx_mask, axis=0))[0]
            vertical_indicies = np.where(np.any(idx_mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                # x2 and y2 should not be part of the box. Increment by 1.
                x2 += 1
                y2 += 1
                
                if self.opt.ext_recept_field:
                    x1 = x1 - 20 if x1 >= 20 else 0
                    y1 = y1 - 20 if y1 >= 20 else 0
                    x2 = x2 + 20 if x2 < (self.opt.width - 20) else (self.opt.width-1)
                    y2 = y2 + 20 if y2 < (self.opt.height - 20) else (self.opt.height-1)

                    # RoI_width = x2 - x1
                    # RoI_height = y2 - y1
                    
                    # # pad the RoI with ratio 1.5
                    # RoI_width_pad = RoI_width * 0.15
                    # RoI_height_pad = RoI_height * 0.15

                    # x1 = 0 if x1 - RoI_width_pad <= 0 else x1 - RoI_width_pad
                    # y1 = 0 if y1 - RoI_height_pad <= 0 else y1 - RoI_height_pad
                    # x2 = self.opt.width if x2 + RoI_width_pad >= self.opt.width else x2 + RoI_width_pad
                    # y2 = self.opt.height if y2 + RoI_height_pad >= self.opt.height else y2 + RoI_height_pad

            else:
                # No mask for this instance
                x1, y1, x2, y2 = 0, 0, self.opt.width-1, self.opt.height-1

            ins_warp_bbox.append(torch.Tensor([[np.float(x1)/32.0, np.float(y1)/32.0, np.float(x2)/32.0, np.float(y2)/32.0]]).to(self.device))

        # [[1, 4]*bs]
        # print(x1/32, y1/32, x2/32, y2/32)
        return ins_warp_bbox


    def get_sem_ins(self, sem_ins_path):
        sem_ins = np.load(sem_ins_path)
        
        return sem_ins

    def omit_small_RoI_pad(self, x_1, y_1, x_2, y_2, width, height):
        RoI_width = x_2 - x_1
        RoI_height = y_2 - y_1

        # pad the RoI with ratio 1.3
        RoI_width_pad = RoI_width * 0.15
        RoI_height_pad = RoI_height * 0.15

        #
        # (x1, y1) ----------------------------
        # |                |                  |
        # |                |                  |
        # |---------RoI: bbox of the Mask ----|
        # |                |                  |
        # |                |                   |
        # ------------------------------(x2, y2)

        if RoI_width * RoI_height < 10*10:
            # if the obj is too small, use the entire img
            x_1 = 0
            y_1 = 0
            x_2 = width
            y_2 = height
        else:
            x_1 = 0 if x_1 - RoI_width_pad <= 0 else x_1 - RoI_width_pad
            y_1 = 0 if y_1 - RoI_height_pad <= 0 else y_1 - RoI_height_pad
            x_2 = width if x_2 + RoI_width_pad >= width else x_2 + RoI_width_pad
            y_2 = height if y_2 + RoI_height_pad >= height else y_2 + RoI_height_pad

        return x_1, y_1, x_2, y_2

    def get_ins_bbox(self, ins_txt_path, ratio_w, ratio_h, width, height):

        # method 1: extract bbox from ins data
        # method 2: load bbox from local disk
        with warnings.catch_warnings():
            # if there is no bbox, the txt file is empty.
            warnings.simplefilter("ignore")
            ins_bbox_mat = np.loadtxt(ins_txt_path)

        K_num = 5 # assume the maximum k+1=4+1=5 (including the bg)
        if len(ins_bbox_mat) > 0:
            if len(ins_bbox_mat.shape) == 1:
                # if there is only one obj
                ins_bbox_mat = np.expand_dims(ins_bbox_mat, 0)
                # (4,) -> (1,4)

            RoI_bbox = []
            if len(ins_bbox_mat) >= K_num-1: # e.g. 4 >= 4 or 5 >= 4
                select_K = K_num-1 # select_K = 4
            else: # 3 < 4
                select_K = len(ins_bbox_mat)

            for i in range(select_K): # only K obj instances are included, K=4
                x_1 = int(ins_bbox_mat[i, 0] * ratio_w)
                y_1 = int(ins_bbox_mat[i, 1] * ratio_h)
                x_2 = int(ins_bbox_mat[i, 2] * ratio_w)
                y_2 = int(ins_bbox_mat[i, 3] * ratio_h)

                x_1, y_1, x_2, y_2 = self.omit_small_RoI_pad(x_1, y_1, x_2, y_2, width, height)
                RoI_bbox.append([x_1/32, y_1/32, x_2/32, y_2/32])

            if len(ins_bbox_mat) < K_num-1:
                x_1 = 0
                y_1 = 0
                x_2 = width/32
                y_2 = height/32
                for i in range(K_num-1-len(ins_bbox_mat)):
                    RoI_bbox.append([x_1, y_1, x_2, y_2])
        else:
            RoI_bbox= []
            x_1 = 0
            y_1 = 0
            x_2 = width/32
            y_2 = height/32

            for i in range(K_num-1):
                RoI_bbox.append([x_1, y_1, x_2, y_2])

        # (4, 4)
        return np.asarray(RoI_bbox)

    def extract_flow(self, pix_coords):
        new_pix_coords = pix_coords.clone()
        # [-1, 1] -> [0, 1] -> [0, w], [b, h, w, 2]
        new_pix_coords = new_pix_coords / 2.0 + 0.5

        new_pix_coords[:, :, :, 0] *= (new_pix_coords.shape[2]-1) # w
        new_pix_coords[:, :, :, 1] *= (new_pix_coords.shape[1]-1) # h

        xx, yy = np.meshgrid(np.arange(0, self.opt.width), np.arange(0, self.opt.height))
        meshgrid = np.transpose(np.stack([xx,yy], axis=-1), [2,0,1]) # [2,h,w]
        cur_pix_coords = torch.from_numpy(meshgrid).unsqueeze(0).repeat(self.opt.batch_size,1,1,1).float().to(self.device) # [b,2,h,w]
        cur_pix_coords = cur_pix_coords.permute(0, 2, 3, 1) # [b,h,w,2]

        flow_pred = new_pix_coords - cur_pix_coords

        return flow_pred

    def add_flow(self, ins_flow, ins_base_flow):
        # 192, 640, 2
        # index: 0 ~ 192*640-1
        # use index to get index_2 through ins_flow (T_dynamic)
        # use index_2 to get index_3 through ins_base_flow (T_static), which is the final flow
        height, width, _ = ins_flow.shape
        first_flow = ins_flow.view(-1, 2).clone() # [192*640, 2]
        second_flow = ins_base_flow.view(-1, 2).clone() # [192*640, 2]

        first_flow_x = first_flow[:, 0] # [192*640,]
        first_flow_y = first_flow[:, 1] # [192*640,]
        second_flow_x = second_flow[:, 0] # [192*640,]
        second_flow_y = second_flow[:, 1] # [192*640,]

        # index = torch.arange(0, first_flow.shape[0]).to(self.device)
        # index2_x_delta = torch.gather(first_flow_x, 0, index)#.to(dtype=torch.int64)
        # index2_y_delta = torch.gather(first_flow_y, 0, index)#.to(dtype=torch.int64)

        index2_x_delta = first_flow_x.to(dtype=torch.int64)
        index2_y_delta = first_flow_y.to(dtype=torch.int64)

        # xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        # meshgrid = np.stack([xx,yy], axis=-1) # [h,w,2]
        # xx, yy = torch.meshgrid(torch.arange(0, width), torch.arange(0, height))
        # HIGHLIGHT: 
        ori_yy, ori_xx = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        ori_xx = torch.reshape(ori_xx, (-1, 1)).to(self.device)
        ori_yy = torch.reshape(ori_yy, (-1, 1)).to(self.device)

        index2_x = torch.clamp(ori_xx.squeeze() + index2_x_delta, 0, width-1)
        index2_y = torch.clamp(ori_yy.squeeze() + index2_y_delta, 0, height-1)
        index2 = index2_y * width + index2_x

        final_flow_x_delta = torch.gather(second_flow_x, 0, index2) #.to(dtype=torch.int64)
        final_flow_y_delta = torch.gather(second_flow_y, 0, index2) #.to(dtype=torch.int64)

        final_flow_x = first_flow_x + final_flow_x_delta # torch.clamp(index2_x_delta + final_flow_x_delta, 0, width-1).to(dtype=torch.float)
        final_flow_y = first_flow_y + final_flow_y_delta # torch.clamp(index2_y_delta + final_flow_y_delta, 0, height-1).to(dtype=torch.float)

        final_flow = torch.cat([final_flow_x.unsqueeze(-1), final_flow_y.unsqueeze(-1)], dim=-1)
        final_flow = final_flow.view(height, width, -1)
        return final_flow

    def compute_IOU(self, mask1, mask2):
        """
        mask1: b, 1, h, w
        """
        inter = mask1 * mask2 # b,
        outer = 1 - (1-mask1) * (1-mask2) # b,
        IOU = inter.sum([2, 3]).float() * 1.0 / (outer.sum([2, 3]).float()+1e-3) # b,
        return IOU
    def compute_outer(self, mask1, mask2):
        """
        mask1: b, 1, h, w
        """
        outer = 1 - (1-mask1) * (1-mask2) # b,
        return outer

    def compute_EPE_map(self, tu, tv, u, v, mask, ru = None, rv = None, return_epe_map=False):
        tau = [3, 0.05]
        stu = tu[:]
        stv = tv[:]
        su = u[:]
        sv = v[:]
        smask = mask[:]

        ind_valid = (smask != 0)
        epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
        
        # epe = epe[ind_valid]
        epe = epe * smask
        return epe
    
    def compute_flow_diff(self, input_flow, inputs):
        if self.opt.dataset == 'drivingstereo_eigen':
            # TODO: !!!
            return np.zeros_like(inputs["gt_occ_flow"][0, :, :, 0].cpu().numpy())
        else:
            dst_h = inputs["gt_occ_flow"][0, :, :, 0].shape[0]
            dst_w = inputs["gt_occ_flow"][0, :, :, 0].shape[1]
            
            resized_flow = fl.resize_flow(input_flow[0].cpu().numpy(), dst_w, dst_h)
            flow_diff = self.compute_EPE_map(inputs["gt_occ_flow"][0, :, :, 0].cpu().numpy(), inputs["gt_occ_flow"][0, :, :, 1].cpu().numpy(), 
                                                            resized_flow[:, :, 0], resized_flow[:, :, 1], 
                                                            inputs["gt_occ_flow"][0, :, :, 2].cpu().numpy())
            return flow_diff

    def compute_flow_EPE(self, gt_noc_flow, gt_occ_flow, pred_flow, flow_type, ins_dir, idx):
        valid_class = [1,2,3,4,6,7,8,16,17,18,19,20,21,22,23,24]
        
        # resize pred_flow to the same size as gt_flow
        dst_h = gt_noc_flow.shape[0]
        dst_w = gt_noc_flow.shape[1]
        # print(pred_flow.shape)
        # print(np.array(pred_flow))
        # input()
        pred_flow = fl.resize_flow(pred_flow, dst_w, dst_h)

        if flow_type == "all":
            pass
        elif flow_type == "fg":
            # print(os.path.join(args.ins_dir, '%.6d_10.npy' % idx))
            dynamic_fg_ins = np.load(os.path.join(ins_dir, idx+'_10.npy'))[:,:,0]
            inputs["img_1_ins"] = img_1_ins

            objs = [dynamic_fg_ins==i for i in valid_class]
            fg_mask = (np.expand_dims(np.sum(objs, axis=0),2) * 1.0)

            gt_noc_flow *= fg_mask
            gt_occ_flow *= fg_mask
            pred_flow *= fg_mask

        elif flow_type == "bg":
            dynamic_fg_ins = np.load(os.path.join(ins_dir, idx+'_10.npy'))[:,:,0]
            objs = [dynamic_fg_ins==i for i in valid_class]
            fg_mask = (1 - np.expand_dims(np.sum(objs, axis=0),2) * 1.0)

            gt_noc_flow *= fg_mask
            gt_occ_flow *= fg_mask
            pred_flow *= fg_mask

        # evaluation
        try:
            (single_noc_epe, single_noc_acc) = fl.evaluate_kitti_flow(gt_noc_flow, pred_flow, None)
        except Exception as e:
            single_noc_epe = 0
            single_noc_acc = 1

        try:
            (single_occ_epe, single_occ_acc) = fl.evaluate_kitti_flow(gt_occ_flow, pred_flow, None)
        except Exception as e:
            single_occ_epe = 0
            single_occ_acc = 1

        return single_noc_epe, single_occ_epe
 
    def add_non_rigid_motion(self, points, non_rigid_motion_map,
                            max_x=3.3, max_y=3.3, max_z=3.3, 
                            min_depth=None, max_depth=None):

        # cam_points: bs, 4, 122880
        # non_rigid_motion_map: bs, 3, 192, 640
        cam_points = points.clone() #[:, :3, :] # bs, 3, 122880

        delta_x_inv, delta_y_inv, delta_z_inv = non_rigid_motion_map

        if min_depth is not None:
            delta_x_inv_new = torch.zeros_like(delta_x_inv)
            delta_y_inv_new = torch.zeros_like(delta_y_inv)
            delta_z_inv_new = torch.zeros_like(delta_z_inv)

            delta_x_inv_new[delta_x_inv>=0] = torch.clamp(delta_x_inv[delta_x_inv>=0], min_depth, max_depth)
            delta_y_inv_new[delta_y_inv>=0] = torch.clamp(delta_y_inv[delta_y_inv>=0], min_depth, max_depth)
            delta_z_inv_new[delta_z_inv>=0] = torch.clamp(delta_z_inv[delta_z_inv>=0], min_depth, max_depth)

            delta_x_inv_new[delta_x_inv<0] = torch.clamp(delta_x_inv[delta_x_inv<0], -max_depth, -min_depth)
            delta_y_inv_new[delta_y_inv<0] = torch.clamp(delta_y_inv[delta_y_inv<0], -max_depth, -min_depth)
            delta_z_inv_new[delta_z_inv<0] = torch.clamp(delta_z_inv[delta_z_inv<0], -max_depth, -min_depth)

            delta_x = 1.0 / delta_x_inv_new
            delta_y = 1.0 / delta_y_inv_new
            delta_z = 1.0 / delta_z_inv_new
        else:
            delta_x = delta_x_inv * max_x
            delta_y = delta_y_inv * max_y
            delta_z = delta_z_inv * max_z

        # step2: reshape
        delta_x = delta_x.view(self.opt.batch_size, 1, -1)
        delta_y = delta_y.view(self.opt.batch_size, 1, -1)
        delta_z = delta_z.view(self.opt.batch_size, 1, -1)

        # step3: add
        cam_points[:, 0, :] += delta_x[:, 0, :]
        cam_points[:, 1, :] += delta_y[:, 0, :]
        cam_points[:, 2, :] += delta_z[:, 0, :]

        # ones = nn.Parameter(torch.ones(self.opt.batch_size, 1, self.opt.height * self.opt.width),
                            # requires_grad=False).to(self.device)
        # cam_points = torch.cat([cam_points, ones], 1)

        return cam_points

    def filter_mask(self, mask):
        if self.opt.filter_mask:
            newmask = torch.ones_like(mask)
            newmask = newmask * (mask >= 0.5).float()
            return newmask
        else:
            return mask
    
    def compute_ins_pose_flow(self, cam_points0_pred_stage2, T_static, ins_pix_coords, scale=0):
        # FIXME: 
        # flow_ins_pix_coords = self.project_3d[scale](cam_points, K, T_total)
        # ins_pose_flow_pred = self.extract_flow(ins_pix_coords)
        ins_base_pix_coords = self.project_3d[scale](cam_points0_pred_stage2, self.K[scale], T_static)
        ins_base_flow = self.extract_flow(ins_base_pix_coords)
        ins_pose_flow = self.extract_flow(ins_pix_coords)
        ins_pose_flow_pred = self.add_flow(ins_pose_flow[0], ins_base_flow[0]).unsqueeze(0)

        return ins_pose_flow_pred
    
    
    # def predict_rigid_flow_with_ins_flow_sc_bk(self, inputs, cam_T_cam, depth0,
    #                                     inv_K_dict, K_dict, rigid_flow):
    #     # some definitions
    #     scale = 0
    #     inv_K = inv_K_dict[scale]
    #     K = K_dict[scale]
    #     img0 = inputs["tensorImg1"]
    #     img1 = inputs["tensorImg2"]
    #     mask0 = torch.sum(inputs["img_1_ins"][:, 1:, :, :], 1).unsqueeze(1).float()
    #     mask1 = torch.sum(inputs["img_2_ins"][:, 1:, :, :], 1).unsqueeze(1).float()
    #     outputs = dict()
        
    #     # step2: compute pix_coords of img0_pred and flow
    #     T_static = cam_T_cam
    #     cam_points = self.backproject_depth[scale](
    #         depth0, inv_K) # cam_points of frame 0, [12, 4, 122880]
    #     pix_coords = self.project_3d[scale](
    #         cam_points, K, T_static)
            
    #     img0_pred = F.grid_sample(img1, pix_coords)
    #     mask0_pred = F.grid_sample(mask1, pix_coords)
    #     mask0_pred = self.filter_mask(mask0_pred)
    #     # warp semantic image
    #     if self.opt.SIG:
    #         img0_pred_sem = F.grid_sample(inputs["img_2_sem"], pix_coords)
    #         img0_pred_edge = F.grid_sample(inputs["img_2_edge"], pix_coords)
    #         disp_input0_pred = torch.cat([img0_pred, img0_pred_sem, img0_pred_edge], 1)
    #         depth0_pred = self.predict_depth(disp_input0_pred)
    #     else:
    #         depth0_pred = self.predict_depth(img0_pred)
    #     cam_points0_pred = self.backproject_depth[scale](
    #         depth0_pred, inv_K) # cam_points of frame 0, [12, 4, 122880]

    #     # step3: compute image feature and crop ROI feature
    #     img0_feature = self.models["instance_pose_encoder"](img0)[-1] # [bs, 512, 6, 20]
    #     img0_pred_feature = self.models["instance_pose_encoder"](img0_pred)[-1] # [bs, 512, 6, 20]
        
    #     # FIXME: define the base image and mask
    #     img0_pred2_base = torch.zeros([self.opt.batch_size, 3, self.opt.height, self.opt.width]).cuda()   # final image
    #     mask0_pred2_base = torch.zeros([self.opt.batch_size, 1, self.opt.height, self.opt.width]).cuda() # bs, 1, 192, 640
    #     flow0_pred = self.extract_flow(pix_coords)
    #     # FIXME: 
    #     img0_pred2_base = img0_pred.clone()
    #     flow0_pred2_base = flow0_pred.clone() #torch.zeros_like(flow0_pred)
        
    #     if self.opt.use_depth_ordering:
    #         depth0_pred2_base = 80.0 * torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width], requires_grad=True).to(self.device) # bs, 1, 192, 640
            
    #     instance_K_num = inputs["img_2_ins"].shape[1] - 1
    #     for ins_id in range(instance_K_num-1, -1, -1): # benefit to large instance            # step4: use T_static to transform mask of each ins
    #         # step4: use T_static to transform mask of each ins
    #         img1_ins_mask = inputs["img_2_ins"][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
    #         img0_pred_ins_mask = F.grid_sample(img1_ins_mask, pix_coords) #[b, 1, h, w]
    #         img0_pred_ins_mask = self.filter_mask(img0_pred_ins_mask)

    #         # step5: crop ins feature of img0 and img0_pred
    #         img0_pred_ins_bbox = self.extract_bbox_from_mask(img0_pred_ins_mask)
    #         # img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(3,3)) # [b, 512, 3, 3] 
    #         img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(6,20)) # [b, 512, 3, 3]

    #         img0_ins_feature = torchvision.ops.roi_align(img0_feature, img0_pred_ins_bbox, output_size=(6,20))

    #         # step6: input ins_pose_net and predict ins_pose
    #         if self.opt.disable_inspose_invert:
    #             ins_pose_inputs = [img0_ins_feature, img0_pred_ins_feature]
    #         else:
    #             if frame_id < 0:
    #                 ins_pose_inputs = [img0_pred_ins_feature, img0_ins_feature]
    #             else:
    #                 ins_pose_inputs = [img0_ins_feature, img0_pred_ins_feature]
    #         ins_pose_inputs = torch.cat(ins_pose_inputs, 1)
    #         ins_axisangle, ins_translation = self.models["instance_pose"](ins_pose_inputs)

    #         if self.opt.set_y_zero:
    #             ins_translation[:, 0][:, :, 1] = 0
                
    #         if self.opt.disable_inspose_invert:
    #             ins_cam_T_cam = transformation_from_parameters(ins_axisangle[:, 0], ins_translation[:, 0], invert=False)
    #         else:
    #             ins_cam_T_cam = transformation_from_parameters(ins_axisangle[:, 0], ins_translation[:, 0], invert=(frame_id < 0))

    #         # ins_cam_T_cam: b, 4, 4
    #         T_dynamic = ins_cam_T_cam
            
    #         if self.opt.use_depth_ordering:
    #             ins_pix_coords, img_z = self.project_3d[scale](cam_points, K, T_dynamic, return_z=True)
    #         else:
    #             ins_pix_coords = self.project_3d[scale](cam_points, K, T_dynamic)

    #         #step8: predict frame 0 from frame 1 based on T_dynamic and T_static
    #         img0_pred2_ins = F.grid_sample(img0_pred, ins_pix_coords)
    #         img0_pred2_ins_mask = F.grid_sample(img0_pred_ins_mask, ins_pix_coords) # [bs, 1, 192, 640]
    #         img0_pred2_ins_mask = self.filter_mask(img0_pred2_ins_mask)

    #         # ------------------------stage3------------------------
    #         # version 1
    #         # predict instance motion for each instance
    #         # ------------------------------------------------------
    #         if self.opt.instance_motion and self.opt.instance_motion_v1:
    #             # img0_ins_feature
    #             img0_pred2_ins_bbox = self.extract_bbox_from_mask(img0_pred2_ins_mask)
    #             img0_pred2_feature = self.models["instance_pose_encoder"](img0_pred2_ins)[-1] # [bs, 512, 6, 20]
    #             img0_pred2_ins_feature = torchvision.ops.roi_align(img0_pred2_feature, img0_pred2_ins_bbox, output_size=(6,20))

    #             # input ins_pose_net and predict ins_motion
    #             if self.opt.disable_inspose_invert:
    #                 ins_motion_inputs = [img0_ins_feature, img0_pred2_ins_feature]
    #             else:
    #                 if frame_id < 0:
    #                     ins_motion_inputs = [img0_pred2_ins_feature, img0_ins_feature]
    #                 else:
    #                     ins_motion_inputs = [img0_ins_feature, img0_pred2_ins_feature]
                    
    #             # compute non rigid motion
    #             ins_motion_inputs = torch.cat(ins_motion_inputs, 1)
    #             non_rigid_motion_map = self.models["instance_motion"](ins_motion_inputs)
                
                
    #             # add non rigid motion
    #             if self.opt.max_speed is not None:
    #                 # cam_points0_pred_stage3, delta_motion = self.add_non_rigid_motion(cam_points, non_rigid_motion_map, 
    #                                                     # self.opt.max_speed, self.opt.max_speed, self.opt.max_speed)
    #                 cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map, 
    #                                                                     self.opt.max_speed, self.opt.max_speed, self.opt.max_speed)
    #             else:
    #                 cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map)
                    
    #             pix_coords_stage3 = self.project_3d[scale](
    #                 cam_points0_pred_stage3, K, torch.eye(4).to(self.device))
                    
    #             # step5: warp new image
    #             img0_pred2_ins = F.grid_sample(
    #                                     img0_pred2_ins,
    #                                     pix_coords_stage3,
    #                                     padding_mode="border")
    #             img0_pred2_ins_mask = F.grid_sample(
    #                                     img0_pred2_ins_mask,
    #                                     pix_coords_stage3,
    #                                     padding_mode="border")
    #             img0_pred2_ins_mask = self.filter_mask(img0_pred2_ins_mask)
            
    #         #step8.5: use IOU value to filter invalid points
    #         if self.opt.iou_thres is not None:
    #             img0_ins_mask = inputs["img_1_ins"][:, ins_id+1, :, :].unsqueeze(1).float()

    #             ins_IOU = self.compute_IOU(img0_ins_mask, img1_ins_mask) # [b, 1]
    #             IOU_mask = ins_IOU > self.opt.iou_thres # [b, 1]

    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask * IOU_mask.float() # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

    #         #step8.6: use diff between t_pred and t_gt to eliminate relative static area
    #         if self.opt.roi_diff_thres is not None:
    #             roi_abs = torch.abs(img0_pred * img0_pred_ins_mask - img0 * img0_pred_ins_mask)
                
    #             # roi_abs: bs, 3, 192, 640
    #             roi_sum = torch.sum(roi_abs, dim=[1, 2, 3]) # bs,
    #             mask_sum = torch.sum(img0_pred_ins_mask, dim=[1, 2, 3]) # bs,
    #             roi_diff = roi_sum.float() * 1.0 / (mask_sum.float()+1e-3) # bs,

    #             roi_diff = roi_diff.unsqueeze(1) # [bs, 1]
    #             roi_mask = roi_diff > self.opt.roi_diff_thres # [bs, 1]

    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask * roi_mask.float() # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

    #         #step9: predict image and coords
    #         # img0_pred2_base:[bs, 3, 192, 640], img0_pred_ins_mask_new: [bs, 1, 192, 640], ins_pix_coords: [bs, 192, 640, 2]
    #         if self.opt.use_depth_ordering:
    #             # img_z: bs, 1, 192, 640
    #             ins_z = img_z * img0_pred2_ins_mask
    #             ins_z_mean = torch.sum(ins_z, [1, 2, 3]).float() / (torch.sum(img0_pred2_ins_mask, [1, 2, 3]).float()+1e-3)
    #             depth0_pred_mean = torch.sum(depth0_pred2_base*img0_pred2_ins_mask, [1, 2, 3]).float() / (torch.sum(img0_pred2_ins_mask, [1, 2, 3]).float()+1e-3)
    #             insz_less_than_depth = (ins_z_mean<depth0_pred_mean).unsqueeze(1) # bs, 1

    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask * insz_less_than_depth.float() # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

    #             depth0_pred2_base = torch.add(depth0_pred2_base*(1-img0_pred2_ins_mask), img_z*img0_pred2_ins_mask)
            
    #         if self.opt.eval_flow_filter_size:
    #             mask_sum = torch.sum(img0_pred_ins_mask, [2, 3])
    #             mask_valid = (mask_sum >= 4900)
                
    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask * mask_valid.float() # [b, 1x192x640]
    #             img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

    #         img0_pred2_base = torch.add(img0_pred2_base*(1-img0_pred2_ins_mask), img0_pred2_ins*img0_pred2_ins_mask)
    #         mask0_pred2_base = torch.add(mask0_pred2_base*(1-img0_pred2_ins_mask), img0_pred2_ins_mask)
            
    #         # compute flow
    #         ins_pose_flow_pred = self.compute_ins_pose_flow(cam_points0_pred, T_static, ins_pix_coords, scale)
    #         flow0_pred2_base = torch.add(flow0_pred2_base*(1-img0_pred2_ins_mask.permute(0, 2, 3, 1)),
    #             ins_pose_flow_pred*img0_pred2_ins_mask.permute(0, 2, 3, 1))
        
    #     # mask0_pred2_base = torch.sum(inputs["img_1_ins"][:, 1:, :, :], 1).unsqueeze(1).float()
    #     img0_pred2 = mask0_pred2_base * img0_pred2_base + (1-mask0_pred2_base) * img0_pred
    #     # FIXME:
    #     mask0_pred2 = mask0_pred2_base.clone() # + (1-mask0_pred2_base) * mask0_pred
    #     img0_pred_latest = img0_pred2.clone()
    #     mask0_pred_latest = mask0_pred2.clone()
    #     # FIXME: check 
    #     # flow0_pred2 = flow0_pred2_base.clone()
    #     flow0_pred2 = mask0_pred2_base.permute(0, 2, 3, 1) * flow0_pred2_base + (1-mask0_pred2_base.permute(0, 2, 3, 1)) * flow0_pred
        
    #     # save for vis
    #     img0_pred2_visual = mask0 * img0_pred2_base + (1-mask0) * img0_pred
    #     flow0_pred2_visual = mask0.permute(0, 2, 3, 1) * flow0_pred2_base + (1-mask0.permute(0, 2, 3, 1)) * flow0_pred
        
    #     # ------------------------stage3------------------------
    #     # version 2
    #     # predict a motion map for the whole image/instance area
    #     # ------------------------------------------------------
    #     if self.opt.instance_motion and self.opt.instance_motion_v2:
    #         # Change mask to re-define img0_pred2
    #         # img0_pred2 = mask0_pred2_base * img0_pred2_base + (1-mask0_pred2_base) * img0_pred
    #         # mask0_pred2 = mask0_pred2_base
    #         if self.opt.train_filter_warping_error:
    #             # select valid area in stage2
    #             # aslo used in training 
    #             error_pred = torch.sum(torch.abs(img0_pred - img0), 1)
    #             error_pred2 = torch.sum(torch.abs(img0_pred2_base - img0), 1)
    #             mask_valid = (error_pred > error_pred2).unsqueeze(1).float()
    #             img0_pred2 = mask_valid * img0_pred2_base + (1-mask_valid) * img0_pred
    #             mask0_pred2 = mask_valid * mask0_pred2_base + (1-mask_valid) * mask0_pred
                
    #         if self.opt.use_fg_stage3:
    #             img0_feature = self.models["instance_pose_encoder"](img0 * mask0)[-1] # [bs, 512, 6, 20]
    #             img0_pred2_feature = self.models["instance_pose_encoder"](img0_pred2 * mask0_pred2)[-1] # [bs, 512, 6, 20]
    #         else:
    #             img0_pred2_feature = self.models["instance_pose_encoder"](img0_pred2)[-1] # [bs, 512, 6, 20]
                
    #         # input ins_pose_net and predict ins_motion
    #         if self.opt.disable_inspose_invert:
    #             ins_motion_inputs = [img0_feature, img0_pred2_feature]
    #         else:
    #             if frame_id < 0:
    #                 ins_motion_inputs = [img0_pred2_feature, img0_feature]
    #             else:
    #                 ins_motion_inputs = [img0_feature, img0_pred2_feature]
                
    #         # compute non rigid motion
    #         ins_motion_inputs = torch.cat(ins_motion_inputs, 1)
    #         non_rigid_motion_map = self.models["instance_motion"](ins_motion_inputs)
                
    #         # add non rigid motion
    #         if self.opt.max_speed is not None:
    #             cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map, 
    #                                                                 self.opt.max_speed, self.opt.max_speed, self.opt.max_speed)
    #         else:
    #             cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map)
                
    #         pix_coords_stage3 = self.project_3d[scale](
    #             cam_points0_pred_stage3, K, torch.eye(4).to(self.device))
                
    #         # step5: warp new image
    #         img0_pred3_base = F.grid_sample(
    #                             img0_pred2,
    #                             pix_coords_stage3,
    #                             padding_mode="border")
    #         mask0_pred3_base = F.grid_sample(
    #                             mask0_pred2,
    #                             pix_coords_stage3,
    #                             padding_mode="border")
    #         mask0_pred3_base = self.filter_mask(mask0_pred3_base)
                                
    #         # generate a new image for visualization
    #         if self.opt.predict_img_motion:
    #             img0_pred3 = img0_pred3_base
    #             mask0_pred3 = mask0_pred3_base
    #         else:
    #             # img0_pred_final = torch.add(img0_pred2*(1-mask0_pred3), img0_pred3*mask0_pred3)
    #             # mask0_pred_final = torch.add(mask0_pred_final*(1-mask0_pred3), mask0_pred3)
    #             # img0_pred_latest = mask0_pred_final * img0_pred_final + (1-mask0_pred_final) * img0_pred
    #             img0_pred3 = mask0_pred3_base * img0_pred3_base + (1-mask0_pred3_base) * img0_pred2
    #             mask0_pred3 = mask0_pred3_base #+ (1-mask0_pred3_base) * mask0_pred2
    #             img0_pred3_visual = mask0 * img0_pred3_base + (1-mask0) * img0_pred2 # mask0 * img0_pred3 + (1-mask0) * img0_pred
            
    #         img0_pred_latest = img0_pred3.clone()
    #         mask0_pred_latest = mask0_pred3.clone()

    #         # FIXME: compute ins-motion flow
    #         ins_motion_flow = self.extract_flow(pix_coords_stage3)
    #         # ins_motion_flow_pred = self.compute_ins_motion_flow(inputs, img0_pred2, K, inv_K, T_static, T_dynamic, 
    #         #                                                     pix_coords, ins_pix_coords, pix_coords_stage3, scale=0)
    #         flow0_pred3_base = self.add_flow(ins_motion_flow[0], flow0_pred2[0]).unsqueeze(0)
    #         # flow0_pred3_base = self.compute_ins_motion_flow(cam_points0_pred, torch.eye(4).to(self.device), pix_coords_stage3)

    #         flow0_pred3 = mask0_pred3.permute(0, 2, 3, 1) * flow0_pred3_base + (1-mask0_pred3).permute(0, 2, 3, 1) * flow0_pred2
    #         flow0_pred3_visual = mask0.permute(0, 2, 3, 1) * flow0_pred3_base + (1-mask0).permute(0, 2, 3, 1) * flow0_pred2

    #     # FIXME: filter instance with huge error
    #     if self.opt.eval_flow_filter_warping_error:
    #         # select valid area to construct the final flow
    #         # used in evaluation
    #         '''
    #         error_pred = torch.sum(torch.abs(img0_pred - img0), 1).unsqueeze(0)
    #         # TODO: color_new
    #         error_new = torch.sum(torch.abs(img0_pred_latest - img0) , 1).unsqueeze(0)
    #         mask_valid = (error_pred > error_new).float()
    #         flow_mask = mask0 * mask_valid
    #         # flow_mask = mask_valid

    #         '''
    #         error_pred = torch.sum(torch.abs(img0_pred - img0), [1, 2, 3]).unsqueeze(0)
    #         error_new = torch.sum(torch.abs(img0_pred_latest - img0) , [1, 2, 3]).unsqueeze(0)
    #         mask_valid = (error_pred > error_new)
    #         flow_mask = mask0.view(self.opt.batch_size, -1) # [b, 1x192x640]
    #         flow_mask = flow_mask * mask_valid.float() # [b, 1x192x640]
    #         flow_mask = flow_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

    #         flow0_pred2_visual = flow_mask.permute(0, 2, 3, 1) * flow0_pred2_base + (1-flow_mask).permute(0, 2, 3, 1) * flow0_pred
    #         if self.opt.instance_motion and not self.opt.predict_img_motion:
    #             flow0_pred3_visual = flow_mask.permute(0, 2, 3, 1) * flow0_pred3_base + (1-flow_mask).permute(0, 2, 3, 1) * flow0_pred

    #     '''
    #     flow_diff_ours = self.compute_flow_diff(flow0_pred3, inputs)
    #     flow_diff_mono2 = self.compute_flow_diff(flow0_pred, inputs)
    #     outputs["0_tgt_{t}"] = inputs["tensorImg1"]
    #     # outputs["1_src_{t+1}"] = inputs["tensorImg2"]
    #     outputs["1_pred_by_stage1"] = img0_pred
    #     outputs["2_pred_by_stage2"] = img0_pred2
    #     if self.opt.instance_motion:
    #         outputs["3_pred_by_stage3"] = img0_pred3
    #     # outputs["3_pred_by_ours"] = color_new
    #     outputs["4_flow_by_mono2"] = flow0_pred
    #     outputs["5_flow_by_ours"] = flow0_pred3
    #     outputs["6_mask_pred_by_mono2"] = self.filter_mask(F.grid_sample(torch.sum(inputs["img_2_ins"][:, 1:, :, :], 1).unsqueeze(1).float(), pix_coords))
    #     outputs["7_mask_pred_by_ours"] = mask0_pred2_base
    #     outputs["8_flow_diff_mono2"] = flow_diff_mono2
    #     outputs["9_flow_diff_ours"] = flow_diff_ours
    #     outputs["10_gt_noc_flow"] = inputs["gt_noc_flow"]
    #     outputs["11_gt_occ_flow"] = inputs["gt_occ_flow"]
    #     '''

    #     flow_diff = self.compute_flow_diff(flow0_pred, inputs)
    #     if self.opt.instance_motion:
    #         flow_diff2 = self.compute_flow_diff(flow0_pred2_visual, inputs)
    #         flow_diff3 = self.compute_flow_diff(flow0_pred3_visual, inputs)
    #         outputs["img_tgt"] = inputs["tensorImg1"]
    #         outputs["img_src"] = inputs["tensorImg2"]
    #         outputs["mask_tgt"] = mask0
    #         #outputs["flow_tgt_occ"] = inputs["gt_occ_flow"]
    #         # img
    #         outputs["img_stage1"] = img0_pred
    #         outputs["img_stage2"] = img0_pred2_visual
    #         outputs["img_stage3"] = img0_pred3_visual
    #         # flow
    #         outputs["flow_stage1"] = flow0_pred
    #         outputs["flow_stage2"] = flow0_pred2_visual
    #         outputs["flow_stage3"] = flow0_pred3_visual
    #         # mask
    #         outputs["mask_stage1"] = mask0_pred 
    #         outputs["mask_stage2"] = mask0_pred2
    #         outputs["mask_stage3"] = mask0_pred3
    #         # flow_diff
    #         outputs["flow_diff_stage1"] = flow_diff
    #         outputs["flow_diff_stage2"] = flow_diff2
    #         outputs["flow_diff_stage3"] = flow_diff3
    #         # img_diff
    #         outputs["img_diff_stage1"] = mask0_pred - mask1
    #         outputs["img_diff_stage2"] = mask0_pred2 - mask0_pred
    #         outputs["img_diff_stage3"] = mask0_pred3 - mask0_pred2
    #     elif self.opt.instance_pose:
    #         flow_diff2 = self.compute_flow_diff(flow0_pred2_visual, inputs)
    #         outputs["img_tgt"] = inputs["tensorImg1"]
    #         outputs["img_src"] = inputs["tensorImg2"]
    #         #outputs["flow_tgt_occ"] = inputs["gt_occ_flow"]
    #         # img
    #         outputs["img_stage1"] = img0_pred
    #         outputs["img_stage2"] = img0_pred2_visual
    #         # flow
    #         outputs["flow_stage1"] = flow0_pred
    #         outputs["flow_stage2"] = flow0_pred2_visual
    #         # mask
    #         outputs["mask_stage1"] = mask0_pred 
    #         outputs["mask_stage2"] = mask0_pred2
    #         # flow_diff
    #         outputs["flow_diff_stage1"] = flow_diff
    #         outputs["flow_diff_stage2"] = flow_diff2
    #         # img_diff
    #         outputs["img_diff_stage1"] = mask0_pred - mask1
    #         outputs["img_diff_stage2"] = mask0_pred2 - mask0_pred
    #     else:
    #         outputs["img_tgt"] = inputs["tensorImg1"]
    #         # img
    #         outputs["img_stage1"] = img0_pred
    #         # flow
    #         outputs["flow_stage1"] = flow0_pred
    #         # mask
    #         outputs["mask_stage1"] = mask0_pred 
    #         # flow_diff
    #         outputs["flow_diff_stage1"] = flow_diff
    #         # img_diff
    #         outputs["img_diff_stage1"] = mask0_pred - mask1
        
    #     # outpus["delta_motion"] = delta_motion
    #     return outputs

    def transform_mul(self, A, B):
        C = A.clone()
        C[:, :3,3:] = torch.bmm(A[:, :3,:3], B[:, :3,3:]) + A[:, :3,3:]
        C[:, :3,:3] = torch.bmm(A[:, :3,:3], B[:, :3,:3])
        return C

    def predict_rigid_flow_with_ins_flow(self, inputs, cam_T_cam, depth0,
                                        inv_K_dict, K_dict, rigid_flow):
        # some definitions
        scale = 0
        inv_K = inv_K_dict[scale]
        K = K_dict[scale]
        img0 = inputs["tensorImg1"]
        img1 = inputs["tensorImg2"]
        mask0 = torch.sum(inputs["img_1_ins"][:, 1:, :, :], 1).unsqueeze(1).float()
        mask1 = torch.sum(inputs["img_2_ins"][:, 1:, :, :], 1).unsqueeze(1).float()
        outputs = dict()
        
        # step2: compute pix_coords of img0_pred and flow
        T_static = cam_T_cam
        cam_points = self.backproject_depth[scale](
            depth0, inv_K) # cam_points of frame 0, [12, 4, 122880]
        pix_coords = self.project_3d[scale](
            cam_points, K, T_static)
            
        img0_pred = F.grid_sample(img1, pix_coords, padding_mode="border")
        mask0_pred = F.grid_sample(mask1, pix_coords, padding_mode="border")
        mask0_pred = self.filter_mask(mask0_pred)
        # warp semantic image

        # step3: compute image feature and crop ROI feature
        img0_feature = self.models["instance_pose_encoder"](img0)[-1] # [bs, 512, 6, 20]
        img0_pred_feature = self.models["instance_pose_encoder"](img0_pred)[-1] # [bs, 512, 6, 20]
        
        # FIXME: define the base image and mask
        img0_pred2_base = torch.zeros([self.opt.batch_size, 3, self.opt.height, self.opt.width]).cuda()   # final image
        mask0_pred2_base = torch.zeros([self.opt.batch_size, 1, self.opt.height, self.opt.width]).cuda() # bs, 1, 192, 640
        flow0_pred = self.extract_flow(pix_coords)
        # FIXME: 
        flow0_pred2_base = torch.zeros_like(flow0_pred)
        
        if self.opt.use_depth_ordering:
            depth0_pred2_base = 80.0 * torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width], requires_grad=True).to(self.device) # bs, 1, 192, 640
            
        instance_K_num = inputs["img_2_ins"].shape[1] - 1
        T_dynamic_list = dict()
        for ins_id in range(instance_K_num-1, -1, -1): # benefit to large instance            # step4: use T_static to transform mask of each ins
            # step4: use T_static to transform mask of each ins
            img1_ins_mask = inputs["img_2_ins"][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
            img0_pred_ins_mask = F.grid_sample(img1_ins_mask, pix_coords, padding_mode="border") #[b, 1, h, w]
            img0_pred_ins_mask = self.filter_mask(img0_pred_ins_mask)

            # step5: crop ins feature of img0 and img0_pred
            img0_pred_ins_bbox = self.extract_bbox_from_mask(img0_pred_ins_mask)
            # img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(3,3)) # [b, 512, 3, 3] 
            img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(6,20)) # [b, 512, 3, 3]
            img0_ins_feature = torchvision.ops.roi_align(img0_feature, img0_pred_ins_bbox, output_size=(6,20))

            # step6: input ins_pose_net and predict ins_pose
            if self.opt.disable_inspose_invert:
                ins_pose_inputs = [img0_ins_feature, img0_pred_ins_feature]
            else:
                if frame_id < 0:
                    ins_pose_inputs = [img0_pred_ins_feature, img0_ins_feature]
                else:
                    ins_pose_inputs = [img0_ins_feature, img0_pred_ins_feature]
            ins_pose_inputs = torch.cat(ins_pose_inputs, 1)
            ins_axisangle, ins_translation = self.models["instance_pose"](ins_pose_inputs)

            if self.opt.set_y_zero:
                ins_translation[:, 0][:, :, 1] = 0
                
            if self.opt.disable_inspose_invert:
                ins_cam_T_cam = transformation_from_parameters(ins_axisangle[:, 0], ins_translation[:, 0], invert=False)
            else:
                ins_cam_T_cam = transformation_from_parameters(ins_axisangle[:, 0], ins_translation[:, 0], invert=(frame_id < 0))

            # ins_cam_T_cam: b, 4, 4
            T_dynamic = ins_cam_T_cam 
            T_dynamic_list[ins_id]=T_dynamic
            
            ins_pix_coords, img_z_stage2 = self.project_3d[scale](cam_points, K, torch.bmm(T_static, T_dynamic), return_z=True)

            #step8: predict frame 0 from frame 1 based on T_dynamic and T_static
            # img0_pred2_ins = F.grid_sample(img0_pred, ins_pix_coords)
            # img0_pred2_ins_mask = F.grid_sample(img0_pred_ins_mask, ins_pix_coords) # [bs, 1, 192, 640]
            img0_pred2_ins = F.grid_sample(img1, ins_pix_coords, padding_mode="border")
            img0_pred2_ins_mask = F.grid_sample(img1_ins_mask, ins_pix_coords, padding_mode="border") # [bs, 1, 192, 640]
            img0_pred2_ins_mask = self.filter_mask(img0_pred2_ins_mask)

            #step8.6: use diff between t_pred and t_gt to eliminate relative static area
            if self.opt.roi_diff_thres is not None:
                roi_abs = torch.abs(img0_pred * img0_pred_ins_mask - img0 * img0_pred_ins_mask)
                
                # roi_abs: bs, 3, 192, 640
                roi_sum = torch.sum(roi_abs, dim=[1, 2, 3]) # bs,
                mask_sum = torch.sum(img0_pred_ins_mask, dim=[1, 2, 3]) # bs,
                roi_diff = roi_sum.float() * 1.0 / (mask_sum.float()+1e-3) # bs,

                roi_diff = roi_diff.unsqueeze(1) # [bs, 1]
                roi_mask = roi_diff > self.opt.roi_diff_thres # [bs, 1]

                img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
                img0_pred2_ins_mask = img0_pred2_ins_mask * roi_mask.float() # [b, 1x192x640]
                img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

            #step9: predict image and coords
            # img0_pred2_base:[bs, 3, 192, 640], img0_pred_ins_mask_new: [bs, 1, 192, 640], ins_pix_coords: [bs, 192, 640, 2]
            if self.opt.use_depth_ordering:
                # img_z: bs, 1, 192, 640
                ins_z = img_z_stage2 * img0_pred2_ins_mask
                ins_z_mean = torch.sum(ins_z, [1, 2, 3]).float() / (torch.sum(img0_pred2_ins_mask, [1, 2, 3]).float()+1e-3)
                depth0_pred_mean = torch.sum(depth0_pred2_base*img0_pred2_ins_mask, [1, 2, 3]).float() / (torch.sum(img0_pred2_ins_mask, [1, 2, 3]).float()+1e-3)
                insz_less_than_depth = (ins_z_mean<depth0_pred_mean).unsqueeze(1) # bs, 1

                img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
                img0_pred2_ins_mask = img0_pred2_ins_mask * insz_less_than_depth.float() # [b, 1x192x640]
                img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

                depth0_pred2_base = torch.add(depth0_pred2_base*(1-img0_pred2_ins_mask), img_z_stage2*img0_pred2_ins_mask)
            
            if self.opt.eval_flow_filter_warping_error_stage2:
                error_pred = torch.sum(torch.abs(img0_pred*img0_pred2_ins_mask - img0*img0_pred2_ins_mask), [1, 2, 3]).unsqueeze(0)
                error_new = torch.sum(torch.abs(img0_pred2_ins*img0_pred2_ins_mask - img0*img0_pred2_ins_mask) , [1, 2, 3]).unsqueeze(0)
                mask_valid = (error_pred > error_new)
                img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
                img0_pred2_ins_mask = img0_pred2_ins_mask * mask_valid.float() # [b, 1x192x640]
                img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)
            
            mask0_pred2_base = torch.add(mask0_pred2_base*(1-img0_pred2_ins_mask), img0_pred2_ins_mask)
            if self.opt.eval_flow_mask_outer:
                cur_img_mask = self.compute_outer(img0_pred2_ins_mask, img0_pred_ins_mask).clone()
                img0_pred2_base = torch.add(img0_pred2_base*(1-cur_img_mask), img0_pred2_ins*cur_img_mask)
            else:    
                img0_pred2_base = torch.add(img0_pred2_base*(1-img0_pred2_ins_mask), img0_pred2_ins*img0_pred2_ins_mask)
            # compute flow
            ins_pose_flow_pred = self.extract_flow(ins_pix_coords)
            flow0_pred2_base = torch.add(flow0_pred2_base*(1-img0_pred2_ins_mask.permute(0, 2, 3, 1)),
                ins_pose_flow_pred*img0_pred2_ins_mask.permute(0, 2, 3, 1))
        
        mask0_pred2 = mask0_pred2_base.clone() # + (1-mask0_pred2_base) * mask0_pred
        if self.opt.eval_flow_mask_outer:
            cur_img_mask = self.compute_outer(mask0_pred2_base, mask0_pred).clone()
            img0_pred2 = cur_img_mask * img0_pred2_base + (1-cur_img_mask) * img0_pred
        else:    
            img0_pred2 = mask0_pred2_base * img0_pred2_base + (1-mask0_pred2_base) * img0_pred
        flow0_pred2 = mask0_pred2_base.permute(0, 2, 3, 1) * flow0_pred2_base + (1-mask0_pred2_base.permute(0, 2, 3, 1)) * flow0_pred
        
        # save for vis
        img0_pred2_visual = mask0 * img0_pred2_base + (1-mask0) * img0_pred
        flow0_pred2_visual = mask0.permute(0, 2, 3, 1) * flow0_pred2_base + (1-mask0.permute(0, 2, 3, 1)) * flow0_pred
        
        # ------------------------stage3------------------------
        # version 2
        # predict a motion map for the whole image/instance area
        # ------------------------------------------------------
        if self.opt.instance_motion and self.opt.instance_motion_v2:
            # Change mask to re-define img0_pred2
            img0_pred2_feature = self.models["instance_pose_encoder"](img0_pred2)[-1] # [bs, 512, 6, 20]
                
            # input ins_pose_net and predict ins_motion
            if self.opt.disable_inspose_invert:
                ins_motion_inputs = [img0_feature, img0_pred2_feature]
            else:
                if frame_id < 0:
                    ins_motion_inputs = [img0_pred2_feature, img0_feature]
                else:
                    ins_motion_inputs = [img0_feature, img0_pred2_feature]
                
            # compute non rigid motion
            ins_motion_inputs = torch.cat(ins_motion_inputs, 1)
            non_rigid_motion_map = self.models["instance_motion"](ins_motion_inputs)
             
            # add non rigid motion
            cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map)#,
                                                                # self.opt.min_depth, self.opt.max_depth)
            
            img0_pred3_base = torch.zeros_like(img0_pred, requires_grad=True)   # final image
            mask0_pred3_base = torch.zeros([self.opt.batch_size, 1, self.opt.height, self.opt.width], requires_grad=True).to(self.device) # bs, 1, 192, 640
            flow0_pred3_base = torch.zeros_like(flow0_pred2_base, requires_grad=True)
            
            for ins_id in range(instance_K_num-1, -1, -1):
                T_dynamic = T_dynamic_list[ins_id]
                img1_ins_mask = inputs["img_2_ins"][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
                pix_coords_stage3 = self.project_3d[scale](
                    cam_points0_pred_stage3, K, torch.bmm(T_static, T_dynamic))
                # step5: warp to a new image
                img0_pred3_ins = F.grid_sample(
                                        img1,
                                        pix_coords_stage3,
                                        padding_mode="border")
                img0_pred3_ins_mask = F.grid_sample(
                                        img1_ins_mask,
                                        pix_coords_stage3,
                                        padding_mode="border")
                img0_pred3_ins_mask = self.filter_mask(img0_pred3_ins_mask)
                ins_motion_flow = self.extract_flow(pix_coords_stage3)
            
                img0_pred3_base = torch.add(img0_pred3_base*(1-img0_pred3_ins_mask), img0_pred3_ins*img0_pred3_ins_mask)
                mask0_pred3_base = torch.add(mask0_pred3_base*(1-img0_pred3_ins_mask), img0_pred3_ins_mask)
                mask0_pred3_base = self.filter_mask(mask0_pred3_base)
                flow0_pred3_base = torch.add(flow0_pred3_base*(1-img0_pred3_ins_mask.permute(0, 2, 3, 1)),
                                                ins_motion_flow*img0_pred3_ins_mask.permute(0, 2, 3, 1))

            # generate a new image for visualization
            mask0_pred3 = mask0_pred3_base.clone() 
            if self.opt.eval_flow_mask_outer:
                cur_img_mask = self.compute_outer(mask0_pred3, mask0_pred2).clone()
                img0_pred3 = cur_img_mask * img0_pred3_base + (1-cur_img_mask) * img0_pred2
            else:
                # False
                img0_pred3 = mask0_pred3_base * img0_pred3_base + (1-mask0_pred3_base) * img0_pred2
            flow0_pred3 = mask0_pred3_base.permute(0, 2, 3, 1) * flow0_pred3_base + (1-mask0_pred3_base.permute(0, 2, 3, 1)) * flow0_pred2

            img0_pred3_visual = mask0 * img0_pred3_base + (1-mask0) * img0_pred2 
            flow0_pred3_visual = mask0.permute(0, 2, 3, 1) * flow0_pred3_base + (1-mask0.permute(0, 2, 3, 1)) * flow0_pred2

            if self.opt.eval_flow_filter_warping_error_stage3:
                # select valid area to construct the final flow
                # used in evaluation
                error_pred = torch.sum(torch.abs(img0_pred2 - img0), [1, 2, 3]).unsqueeze(0)
                error_new = torch.sum(torch.abs(img0_pred3 - img0) , [1, 2, 3]).unsqueeze(0)
                mask_valid = (error_pred > error_new)
                flow_mask = mask_valid.float() # [b, 1x192x640]
                flow0_pred3 = flow_mask * flow0_pred3 + (1-flow_mask) * flow0_pred2

        flow_diff = self.compute_flow_diff(flow0_pred, inputs)
        # outputs["depth0"] = depth0
        if self.opt.instance_motion:
            flow_diff2 = self.compute_flow_diff(flow0_pred2, inputs)
            flow_diff3 = self.compute_flow_diff(flow0_pred3, inputs)
            outputs["img_tgt"] = inputs["tensorImg1"]
            outputs["img_src"] = inputs["tensorImg2"]
            outputs["mask_tgt"] = mask0
            #outputs["flow_tgt_occ"] = inputs["gt_occ_flow"]
            # img
            outputs["img_stage1"] = img0_pred
            outputs["img_stage2"] = img0_pred2
            outputs["img_stage3"] = img0_pred3
            # flow
            outputs["flow_stage1"] = flow0_pred
            outputs["flow_stage2"] = flow0_pred2
            outputs["flow_stage3"] = flow0_pred3
            # mask
            outputs["mask_stage1"] = mask0_pred 
            outputs["mask_stage2"] = mask0_pred2
            outputs["mask_stage3"] = mask0_pred3
            # flow_diff
            outputs["flow_diff_stage1"] = flow_diff
            outputs["flow_diff_stage2"] = flow_diff2
            outputs["flow_diff_stage3"] = flow_diff3
            # img_diff
            outputs["img_diff_stage1"] = mask0_pred - mask1
            outputs["img_diff_stage2"] = mask0_pred2 - mask0_pred
            outputs["img_diff_stage3"] = mask0_pred3 - mask0_pred2
            # outputs["img_diff_stage1"] = img0_pred - img1
            # outputs["img_diff_stage2"] = img0_pred2 - img0_pred
            # outputs["img_diff_stage3"] = img0_pred3 - img0_pred2
        elif self.opt.instance_pose:
            flow_diff2 = self.compute_flow_diff(flow0_pred2, inputs)
            outputs["img_tgt"] = inputs["tensorImg1"]
            outputs["img_src"] = inputs["tensorImg2"]
            #outputs["flow_tgt_occ"] = inputs["gt_occ_flow"]
            # img
            outputs["img_stage1"] = img0_pred
            outputs["img_stage2"] = img0_pred2
            # flow
            outputs["flow_stage1"] = flow0_pred
            outputs["flow_stage2"] = flow0_pred2
            # mask
            outputs["mask_stage1"] = mask0_pred 
            outputs["mask_stage2"] = mask0_pred2
            # flow_diff
            outputs["flow_diff_stage1"] = flow_diff
            outputs["flow_diff_stage2"] = flow_diff2
            # img_diff
            outputs["img_diff_stage1"] = mask0_pred - mask1
            outputs["img_diff_stage2"] = mask0_pred2 - mask0_pred
        else:
            outputs["img_tgt"] = inputs["tensorImg1"]
            # img
            outputs["img_stage1"] = img0_pred
            # flow
            outputs["flow_stage1"] = flow0_pred
            # mask
            outputs["mask_stage1"] = mask0_pred 
            # flow_diff
            outputs["flow_diff_stage1"] = flow_diff
            # img_diff
            outputs["img_diff_stage1"] = mask0_pred - mask1
        
        # outpus["delta_motion"] = delta_motion
        return outputs

    def predict_flow_img_pair(self, inputs):
        if self.opt.SIG:
            disp_input_1 = torch.cat([inputs["tensorImg1"], inputs["img_1_sem"], inputs["img_1_edge"]], 1)
            disp_input_2 = torch.cat([inputs["tensorImg2"], inputs["img_2_sem"], inputs["img_2_edge"]], 1)

            depth_1 = self.predict_depth(disp_input_1)
            # cam_T_cam = self.predict_poses(disp_input_1, disp_input_2)
        else:
            depth_1 = self.predict_depth(inputs["tensorImg1"])
        
        cam_T_cam = self.predict_poses(inputs["tensorImg1"], inputs["tensorImg2"])

        #if self.opt.predict_delta:
        if self.opt.instance_pose:
            # flow = rigid_flow + non_rigid_flow
            rigid_flow_outputs = self.predict_rigid_flow(cam_T_cam, depth_1, self.inv_K, self.K)
            if self.opt.SIG:
                flow_outputs = self.predict_rigid_flow_with_ins_flow(
                                 inputs, cam_T_cam, depth_1,
                                 self.inv_K, self.K,
                                 rigid_flow_outputs["flow_stage1"])
            else:
                flow_outputs = self.predict_rigid_flow_with_ins_flow(
                                            inputs, cam_T_cam, depth_1,
                                            self.inv_K, self.K,
                                            rigid_flow_outputs["flow_stage1"])
        else:
            #FIXME: whether use ins_pose_net
            # only compute rigid_flow
            flow_outputs = self.predict_rigid_flow(cam_T_cam, depth_1, self.inv_K, self.K, inputs["tensorImg2"])

        return flow_outputs

    def load_img(self, input_dir, input_file, img_ext='png', idx=None):
        input_path_1 = os.path.join(input_dir, input_file+"_10."+img_ext)
        input_path_2 = os.path.join(input_dir, input_file+"_11."+img_ext)

        #[h, w, 3] -> [3, h, w]->[1, 3, h, w]
        resize_func = transforms.Resize((self.opt.height, self.opt.width),
                                        interpolation=PIL.Image.ANTIALIAS)

        img1 = resize_func(PIL.Image.open(input_path_1).convert('RGB'))
        img2 = resize_func(PIL.Image.open(input_path_2).convert('RGB'))

        # [b, 3, h, w]
        tensorImg1 = transforms.ToTensor()(img1).unsqueeze(0)
        tensorImg2 = transforms.ToTensor()(img2).unsqueeze(0)

        # load gt flow 
        if self.opt.dataset == 'drivingstereo_eigen':
            gt_occ_fn = os.path.join('dataset/ds_scene_flow_flowgt/'+str(idx)+".png")
            gt_occ_flow = fl.read_flow(gt_occ_fn)
            return tensorImg1, tensorImg2, gt_occ_flow, gt_occ_flow
        elif self.opt.dataset == 'kitti_raw_data_flow':
            gt_noc_fn = os.path.join('dataset/kitti_raw_data_flow/flow_gt/'+input_file+"_10."+img_ext)
            gt_noc_flow = fl.read_flow(gt_noc_fn)
            gt_occ_flow = gt_noc_flow.copy()

            return tensorImg1, tensorImg2, gt_noc_flow, gt_occ_flow
        else:
            gt_noc_fn = os.path.join('dataset/data_scene_flow/training/flow_noc/'+input_file+"_10."+img_ext)
            gt_occ_fn = os.path.join('dataset/data_scene_flow/training/flow_occ/'+input_file+"_10."+img_ext)

            gt_noc_flow = fl.read_flow(gt_noc_fn)
            gt_occ_flow = fl.read_flow(gt_occ_fn)

            return tensorImg1, tensorImg2, gt_noc_flow, gt_occ_flow

    def convert_sem_onehot(self, input_var):
        max_num = int(np.max(input_var))
        sem_tensor = torch.Tensor(np.asarray(input_var))
        K_num = 19
        if max_num+1 <= K_num:
            sem_tensor_one_hot = torch.nn.functional.one_hot(sem_tensor.to(torch.int64), K_num).type(torch.FloatTensor)
        else:
            sem_tensor_one_hot = torch.nn.functional.one_hot(sem_tensor.to(torch.int64), max_num+1).type(torch.FloatTensor)
            sem_tensor_one_hot = sem_tensor_one_hot[:,:,:K_num]

        # sem_seg_one_hot
        return sem_tensor_one_hot.permute(2,0,1).unsqueeze(0)

    def load_sem(self, input_dir, input_file):
        input_path_1 = os.path.join(input_dir, input_file+"_10.npy")
        input_path_2 = os.path.join(input_dir, input_file+"_11.npy")

        if self.opt.dataset == 'kitti_raw_data_flow':
            sem_1 = Image.fromarray(np.uint8(self.get_sem_ins(input_path_1)[:, :, 0])).convert("L")
            sem_2 = Image.fromarray(np.uint8(self.get_sem_ins(input_path_2)[:, :, 0])).convert("L")
        else:
            sem_1 = Image.fromarray(np.uint8(self.get_sem_ins(input_path_1))).convert("L")
            sem_2 = Image.fromarray(np.uint8(self.get_sem_ins(input_path_2))).convert("L")

        sem_1 = sem_1.resize((self.opt.width, self.opt.height), Image.NEAREST)
        sem_2 = sem_2.resize((self.opt.width, self.opt.height), Image.NEAREST)

        sem_1_onehot = self.convert_sem_onehot(sem_1)
        sem_2_onehot = self.convert_sem_onehot(sem_2)

        return sem_1_onehot, sem_2_onehot

    def get_edge(self, ins_id_seg):
        ins_edge_seg = None
        ins_id_seg_edge_gradient = np.gradient(ins_id_seg)
        x = ins_id_seg_edge_gradient[0]
        y = ins_id_seg_edge_gradient[1]

        ins_edge_seg = ((x+y)!=0)*1

        return ins_edge_seg

    def convert_ins_onehot(self, input_var):
        max_num = int(np.max(input_var))
        ins_tensor = torch.Tensor(np.asarray(input_var))
        K_num = 5
        if max_num+1 <= K_num:
            ins_tensor_one_hot = torch.nn.functional.one_hot(ins_tensor.to(torch.int64), K_num).type(torch.bool)
        else:
            ins_tensor_one_hot = torch.nn.functional.one_hot(ins_tensor.to(torch.int64), max_num+1).type(torch.bool)

        ins_tensor_one_hot = ins_tensor_one_hot[:,:,:K_num]

        return ins_tensor_one_hot.permute(2,0,1).unsqueeze(0)

    def extract_bbox_ins_edge(self, file_name):
        bbox_path = os.path.join(bbox_dir, file_name.replace("npy","txt"))
        ins_path = os.path.join(ins_dir, file_name)

        ins_seg = self.get_sem_ins(ins_path)
        sig_ins_id_seg = PIL.Image.fromarray(np.uint8(ins_seg[:,:,1])).convert("L")
        ins_width, ins_height = sig_ins_id_seg.size

        sig_ins_id_seg = sig_ins_id_seg.resize((self.opt.width, self.opt.height), PIL.Image.NEAREST)

        ins_id_seg_to_edge = np.expand_dims(self.get_edge(np.asarray(sig_ins_id_seg)), -1)
        ins_edge = torch.Tensor(ins_id_seg_to_edge).permute(2,0,1).unsqueeze(0)

        ins_onehot = self.convert_ins_onehot(sig_ins_id_seg)

        ratio_w = self.opt.width / ins_width
        ratio_h = self.opt.height / ins_height

        # print("ratio_w: ", ratio_w)
        # print("ratio_h: ", ratio_h)

        ins_RoI_bbox = self.get_ins_bbox(bbox_path, ratio_w, ratio_h, self.opt.width, self.opt.height)

        ins_RoI_bbox = torch.Tensor(ins_RoI_bbox).unsqueeze(0) #[bs, k=4 ,4]

        return ins_RoI_bbox, ins_onehot, ins_edge

    def expand_mask(self, ins_warp_mask):
        """ expand mask area
        """
        # ins_warp_mask: [bs, 1, 192, 640]
        mask = ins_warp_mask.squeeze(1)
        # new_ins_warp_mask = torch.zeros_like(ins_warp_mask, requires_grad=True)
        new_ins_warp_mask = ins_warp_mask

        for bs_idx in range(mask.shape[0]):
            # Bounding box.
            idx_mask = mask[bs_idx, :, :].type(torch.uint8)
            horizontal_indicies = torch.where(torch.any(idx_mask, axis=0))[0]
            vertical_indicies = torch.where(torch.any(idx_mask, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                
                RoI_width = x2 - x1
                RoI_height = y2 - y1
                # pad the RoI with ratio 1.5
                RoI_width_pad = RoI_width * 0.15
                RoI_height_pad = RoI_height * 0.15

                x1 = 0 if x1 - RoI_width_pad < 0 else x1 - RoI_width_pad
                y1 = 0 if y1 - RoI_height_pad < 0 else y1 - RoI_height_pad
                x2 = self.opt.width-1 if x2 + RoI_width_pad >= self.opt.width else x2 + RoI_width_pad
                y2 = self.opt.height-1 if y2 + RoI_height_pad >= self.opt.height else y2 + RoI_height_pad

                new_ins_warp_mask[bs_idx, :, y1:y2, x1:x2] = 1.0
        
        return new_ins_warp_mask
    
    def load_bbox_ins_edge(self, bbox_dir, ins_dir, input_file):
        ins_RoI_bbox_1, ins_onehot_1, ins_edge_1 = self.extract_bbox_ins_edge(input_file+"_10.npy")
        ins_RoI_bbox_2, ins_onehot_2, ins_edge_2 = self.extract_bbox_ins_edge(input_file+"_11.npy")

        return ins_RoI_bbox_1, ins_onehot_1, ins_edge_1, ins_RoI_bbox_2, ins_onehot_2, ins_edge_2

    def save_flow(self, flow_outputs, output_dir, input_file):
        if self.opt.instance_motion:
            selected_key = 'flow_stage3'
        elif self.opt.instance_pose:
            selected_key = 'flow_stage2'
        else:
            selected_key = 'flow_stage1'
        
        for key in flow_outputs:
            if selected_key not in key:
                continue

            optical_flow = flow_outputs[key]
            optical_flow = optical_flow[0].cpu().numpy()
            if os.path.exists(os.path.join(output_dir, "npy")) == False:
                os.makedirs(os.path.join(output_dir, "npy"))
            out_path = os.path.join(output_dir, "npy", input_file+".npy")
            np.save(out_path, optical_flow)

    def save_everything(self, flow_outputs, output_dir, input_file, inputs=None):
        # output_dir: output/flow/${MODEL}/weights_${EPOCH}
        if self.opt.dataset == 'drivingstereo_eigen':
            cur_output_dir = output_dir.replace('flow_ds', 'flow').replace('flow', 'flow_ds').replace('flow', 'imgs')
        elif self.opt.dataset == 'kitti_raw_data_flow':
            cur_output_dir = output_dir.replace('flow_raw', 'flow').replace('flow', 'flow_raw').replace('flow', 'imgs')
        else:
            cur_output_dir = output_dir.replace('flow', 'imgs')
        if not os.path.exists(cur_output_dir):
            os.makedirs(cur_output_dir)

        for key in flow_outputs:
            out_path = os.path.join(cur_output_dir, input_file+"_"+key+".png")
            if 'img_tgt' in key:
                cur_img = flow_outputs[key][0].cpu().numpy()
                cur_img = np.transpose(cur_img, (1, 2, 0))
                cv2.imwrite(out_path, cur_img[:, :, ::-1]*255.0)   
                '''
                cur_img = flow_outputs[key][0].cpu().numpy()
                cur_img = np.transpose(cur_img, (1, 2, 0))
                bbox = inputs["img_1_bbox"][0].cpu().numpy()
                i = 1
                x1 = bbox[i][0] * 32.0 
                y1 = bbox[i][1] * 32.0 
                w = (bbox[i][2] - bbox[i][0]) * 32.0 
                h = (bbox[i][3] - bbox[i][1]) * 32.0 
                x2 = x1 + w
                y2 = y1 + h
                x1 = x1 - 20 if x1 >= 20 else 0
                y1 = y1 - 20 if y1 >= 20 else 0
                x2 = x2 + 20 if x2 <= (self.opt.width - 20) else (self.opt.width)
                y2 = y2 + 20 if y2 <= (self.opt.height - 20) else (self.opt.height)
                w = x2 - x1 
                h = y2 - y1
                
                plt.gca().add_patch(plt.Rectangle((x1,y1),w,h, fill=False, edgecolor='r', linewidth=3))
                plt.imshow(cur_img)
                plt.axis('off')
                plt.savefig(out_path, transparent=True)
                plt.close()
                '''
            elif 'img_stage' in key or 'img_src' in key:
                cur_img = flow_outputs[key][0].cpu().numpy()
                cur_img = np.transpose(cur_img, (1, 2, 0))
                cv2.imwrite(out_path, cur_img[:, :, ::-1]*255.0)   
            elif 'flow_stage' in key:
                cur_img = flow_outputs[key][0].cpu().numpy()
                cur_img = fl.flow_to_image(cur_img)
                # cur_img = np.transpose(cur_img, (1, 2, 0))
                cv2.imwrite(out_path, cur_img[:, :, ::-1])
                # print(flow_img.shape)
                # print(cur_img.shape)
                # cv2.imwrite(out_path, cur_img)
            elif 'depth' in key:
                cur_img = flow_outputs[key][0].cpu().numpy()
                # cur_img = np.transpose(cur_img, (1, 2, 0))
                # cv2.imwrite(out_path, cur_img)
                plt.imshow(1.0/cur_img[0], cmap="plasma")
                plt.axis('off')
                plt.savefig(out_path, transparent=True)
                plt.close()
            elif 'img_diff' in key:
                fig = plt.figure(figsize=(8,2))
                cur_img = flow_outputs[key][0].cpu().numpy()
                cur_img = np.transpose(cur_img, (1, 2, 0))
                cur_img = np.mean(cur_img, -1)
                # cur_img = (cur_img + 1.0) / 2.0
                cur_img = (cur_img * 255).astype(int)
                plt.axis('off')
                plt.imshow(cur_img, vmin=-255, vmax=255, cmap="RdBu")

                l = 0.85
                b = 0.1
                w = 0.015
                h = 0.8

                rect = [l,b,w,h] 
                cbar_ax = fig.add_axes(rect) 
                cb = plt.colorbar(cax=cbar_ax)
                cb.set_ticks(np.linspace(-255,255, 5).astype(int))
                # cbar.set_ticklabels( ('160', '180', '200', '220', '240',  '260',  '280',  '300'))
                plt.savefig(out_path, transparent=True)
                plt.close()
            # elif 'flow_diff' in key:
            #     fig = plt.figure(figsize=(8,4))
            #     cur_img = flow_outputs[key]
            #     # cur_img = np.transpose(cur_img, (1, 2, 0))
            #     # cur_img = np.mean(cur_img, -1)
                
            #     plt.axis('off')
            #     plt.imshow(cur_img, vmin=-50, vmax=50, cmap="RdBu")

            #     l = 0.92
            #     b = 0.30
            #     w = 0.015
            #     h = 0.38

            #     rect = [l,b,w,h] 
            #     cbar_ax = fig.add_axes(rect) 
            #     cb = plt.colorbar(cax=cbar_ax)
            #     # plt.colorbar()
            #     plt.savefig(out_path, transparent=True)
            #     plt.close()
            else:
                print(key)
                continue 

            # plt.imshow(cur_img)
            # plt.show()
            # plt.imsave(out_path, cur_img)
            # plt.close()

    def eval_flow(self, flow_outputs, input_file, ins_dir, inputs=None):
        if self.opt.instance_motion:
            selected_key = 'flow_stage3'
        elif self.opt.instance_pose:
            selected_key = 'flow_stage2'
        else:
            selected_key = 'flow_stage1'
        pred_flow = flow_outputs[selected_key]
        fg_single_noc_epe, fg_single_occ_epe = self.compute_flow_EPE(inputs["gt_noc_flow"][0].cpu().numpy(), inputs["gt_occ_flow"][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'fg', ins_dir, input_file)
        bg_single_noc_epe, bg_single_occ_epe = self.compute_flow_EPE(inputs["gt_noc_flow"][0].cpu().numpy(), inputs["gt_occ_flow"][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'bg', ins_dir, input_file)
        all_single_noc_epe, all_single_occ_epe = self.compute_flow_EPE(inputs["gt_noc_flow"][0].cpu().numpy(), inputs["gt_occ_flow"][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'all', ins_dir, input_file)
        # noc_mean_epe = np.mean(noc_epe)
        # noc_mean_acc = np.mean(noc_acc)
        # occ_mean_epe = np.mean(occ_epe)
        # occ_mean_acc = np.mean(occ_acc)

        # print('Mean Noc EPE = %.4f ' % noc_mean_epe)
        # print('Mean Noc ACC = %.4f ' % noc_mean_acc)
        # print('Mean Occ EPE = %.4f ' % occ_mean_epe)
        # print('Mean Occ ACC = %.4f ' % occ_mean_acc)

        return fg_single_noc_epe, fg_single_occ_epe, bg_single_noc_epe, bg_single_occ_epe, all_single_noc_epe, all_single_occ_epe
        
    def visualize_flow(self, flow_outputs, output_dir, input_file, ins_dir, inputs=None):
        # output_dir: output/flow/${MODEL}/weights_${EPOCH}
        if self.opt.dataset == 'drivingstereo_eigen':
            vis_output_dir = output_dir.replace('flow_ds', 'flow').replace('flow', 'flow_ds').replace('flow', 'vis')
        elif self.opt.dataset == 'kitti_raw_data_flow':
            vis_output_dir = output_dir.replace('flow_raw', 'flow').replace('flow', 'flow_raw').replace('flow', 'vis')
        else:
            vis_output_dir = output_dir.replace('flow', 'vis')
        if os.path.exists(vis_output_dir) == False:
            os.makedirs(vis_output_dir)

        output_path = os.path.join(vis_output_dir, input_file+".jpg")
        plt.figure(figsize=(15, 16))

        for i, key in enumerate(flow_outputs):
            if self.opt.instance_motion:
                plt.subplot(6, 3, i+1)
            elif self.opt.instance_pose:
                plt.subplot(6, 2, i+1)
            else:
                plt.subplot(6, 1, i+1)

            if 'flow_diff' in key:
                cur_img = flow_outputs[key]
                plt.imshow(cur_img, vmin=0, vmax=100)
                # plt.colorbar()
            else:
                cur_img = flow_outputs[key][0].cpu().numpy()
                # if 'flow_by' in key:
                #     # evaluate flow, mono2 and ours
                #     pred_flow = flow_outputs[key]
                #     fg_single_noc_epe, fg_single_occ_epe = self.compute_flow_EPE(flow_outputs['10_gt_noc_flow'][0].cpu().numpy(), flow_outputs['11_gt_occ_flow'][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'fg', ins_dir, input_file)
                #     all_single_noc_epe, all_single_occ_epe = self.compute_flow_EPE(flow_outputs['10_gt_noc_flow'][0].cpu().numpy(), flow_outputs['11_gt_occ_flow'][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'all', ins_dir, input_file)
                #     key = '%s: fg: %.2f, %.2f, all: %.2f,%.2f' % (key, fg_single_noc_epe, fg_single_occ_epe, all_single_noc_epe, all_single_occ_epe)
                #     cur_img = fl.flow_to_image(cur_img)
                # elif 'gt_noc_flow' in key:
                #     cur_img = fl.resize_flow(cur_img, self.opt.width, self.opt.height)
                #     cur_img = fl.flow_to_image(cur_img)
                # elif 'gt_occ_flow' in key:
                #     cur_img = fl.resize_flow(cur_img, self.opt.width, self.opt.height)
                #     cur_img = fl.flow_to_image(cur_img)
                # elif 'flow' in key:
                #     cur_img = fl.flow_to_image(cur_img)
                # elif "mask_pred" in key:
                #     cur_img = np.transpose(cur_img, (1, 2, 0))[:,:,0]
                # else:
                #     cur_img = np.transpose(cur_img, (1, 2, 0))
                
                if 'img' in key:
                    cur_img = np.transpose(cur_img, (1, 2, 0))
                    cur_img = np.clip(cur_img, 0, 1)
                elif 'mask' in key:
                    cur_img = np.transpose(cur_img, (1, 2, 0))[:,:,0]
                elif 'flow_tgt' in key:
                    cur_img = fl.resize_flow(cur_img, self.opt.width, self.opt.height)
                    cur_img = fl.flow_to_image(cur_img)
                elif 'flow' in key:
                    # evaluate flow, mono2 and ours
                    pred_flow = flow_outputs[key]
                    if self.opt.dataset == 'drivingstereo_eigen':
                        pass
                    else:
                        fg_single_noc_epe, fg_single_occ_epe = self.compute_flow_EPE(inputs["gt_noc_flow"][0].cpu().numpy(), inputs["gt_occ_flow"][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'fg', ins_dir, input_file)
                        all_single_noc_epe, all_single_occ_epe = self.compute_flow_EPE(inputs["gt_noc_flow"][0].cpu().numpy(), inputs["gt_occ_flow"][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'all', ins_dir, input_file)
                        key = '%s: fg: %.2f, %.2f, all: %.2f,%.2f' % (key, fg_single_noc_epe, fg_single_occ_epe, all_single_noc_epe, all_single_occ_epe)
                    
                    cur_img = fl.flow_to_image(cur_img)

                plt.imshow(cur_img)

            plt.title(key)
            # plt.colorbar()

        #plt.axis('off')
        plt.savefig(output_path)
        plt.close()


if __name__ == '__main__':
    # load Model
    options = MonodepthOptions()
    opts = options.parse()
    evaler = Evaler(opts)
    output_dir = opts.output_dir # output/flow/${MODEL}/weights_${EPOCH}
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    save_flag = False
    vis_flag = True
    eval_flag = False
    save_per_image = False

    if opts.dataset == 'drivingstereo_eigen':
        # driving stereo dataset
        bbox_dir = "dataset/ds_scene_flow_SIG/bbox"
        sem_dir = "dataset/ds_scene_flow_SIG/sem"
        ins_dir = "dataset/ds_scene_flow_SIG/ins"
        input_dir = "dataset/ds_scene_flow/flow_select_150/image_2"
        img_ext = "jpg"
    elif opts.dataset == 'kitti_raw_data_flow':
        bbox_dir = "dataset/kitti_raw_data_flow/bbox"
        sem_dir = "dataset/kitti_raw_data_flow/sem"
        ins_dir = "dataset/kitti_raw_data_flow/ins"
        input_dir = "dataset/kitti_raw_data_flow/img"
        img_ext = "png"
    else:
        # kitti dataset
        bbox_dir = "dataset/data_scene_flow_SIG/bbox"
        sem_dir = "dataset/data_scene_flow_SIG/sem"
        ins_dir = "dataset/data_scene_flow_SIG/ins"
        input_dir = "dataset/data_scene_flow/training/image_2"
        img_ext = "png"

    invalid_next_idx = [246, 316]
    invalid_pre_idx = [18, 25, 68, 82, 142, 153, 175, 200, 225, 268, 298, 327, 397, 422, 517, 525, 556, 591, 617, 641, 671, 676]
    
    if opts.dataset == 'drivingstereo_eigen':
        total_img = 150
    elif opts.dataset == 'kitti_raw_data_flow':
        total_img = 697
    else: 
        total_img = 200
    

    if eval_flag == True:
        all_noc_epe = []
        all_occ_epe = []
        fg_noc_epe = []
        fg_occ_epe = []
        bg_noc_epe = []
        bg_occ_epe = []
    
    with torch.no_grad():
        selected_idx = [18,19,20, 43, 44, 45, 46, 49, 58, 91, 95, 100, 184, 193, 197]
        # selected_idx = [2, 3, 100, 168, 169, 188, 189]
        # selected_idx = [128, 48, 105, 33, 184, 29, 41, 106, 45, 165, 25, 190, 117, 26, 31, 177] # kitti raw data
        #for i in tqdm(range(total_img)):
        for i in selected_idx:
            if opts.dataset == 'kitti_raw_data_flow' and i in invalid_next_idx:
                continue 
        
            # load img pair
            # t1 = timeit.default_timer()
            input_file = str(i).zfill(6)
            inputs = {}
            tensorImg1, tensorImg2, gt_noc_flow, gt_occ_flow = evaler.load_img(input_dir, input_file, img_ext, idx=i)
            
            # t2 = timeit.default_timer()
            img_1_sem, img_2_sem = evaler.load_sem(sem_dir, input_file)
            
            img_1_bbox, img_1_ins, img_1_edge, img_2_bbox, img_2_ins, img_2_edge = \
                evaler.load_bbox_ins_edge(bbox_dir, ins_dir, input_file)

            inputs["tensorImg1"] = tensorImg1
            inputs["tensorImg2"] = tensorImg2
            inputs["img_1_bbox"] = img_1_bbox
            inputs["img_2_bbox"] = img_2_bbox
            inputs["img_1_sem"] = img_1_sem
            inputs["img_2_sem"] = img_2_sem
            inputs["img_1_ins"] = img_1_ins
            inputs["img_1_edge"] = img_1_edge
            inputs["img_2_ins"] = img_2_ins
            inputs["img_2_edge"] = img_2_edge
            inputs["gt_noc_flow"] = torch.Tensor(gt_noc_flow).unsqueeze(0)
            inputs["gt_occ_flow"] = torch.Tensor(gt_occ_flow).unsqueeze(0)
            
            for key, ipt in inputs.items():
                inputs[key] = ipt.to("cuda")
            
            # compute flow in MonoDepth2
            # t3 = timeit.default_timer()
            flow_outputs = evaler.predict_flow_img_pair(inputs)
            # t4 = timeit.default_timer()

            # save flow
            if save_flag == True:
                evaler.save_flow(flow_outputs, output_dir, input_file)

            # visualization
            if vis_flag == True:
                evaler.visualize_flow(flow_outputs, output_dir, input_file, ins_dir, inputs)

            # eval flow
            if eval_flag == True:
                fg_single_noc_epe, fg_single_occ_epe, bg_single_noc_epe, bg_single_occ_epe, all_single_noc_epe, all_single_occ_epe = evaler.eval_flow(flow_outputs, input_file, ins_dir, inputs)
                all_noc_epe.append(all_single_noc_epe)
                all_occ_epe.append(all_single_occ_epe)
                fg_noc_epe.append(fg_single_noc_epe)
                fg_occ_epe.append(fg_single_occ_epe)
                bg_noc_epe.append(bg_single_noc_epe)
                bg_occ_epe.append(bg_single_occ_epe)
            # t5 = timeit.default_timer()
        
            if save_per_image == True:
                evaler.save_everything(flow_outputs, output_dir, input_file, inputs)

        if eval_flag == True:
            all_noc_mean_epe = np.mean(all_noc_epe)
            all_occ_mean_epe = np.mean(all_occ_epe)
            fg_noc_mean_epe = np.mean(fg_noc_epe)
            fg_occ_mean_epe = np.mean(fg_occ_epe)
            bg_noc_mean_epe = np.mean(bg_noc_epe)
            bg_occ_mean_epe = np.mean(bg_occ_epe)

            print(all_noc_mean_epe, all_occ_mean_epe, fg_noc_mean_epe, fg_occ_mean_epe, bg_noc_mean_epe, bg_occ_mean_epe)
            # opts.eval_flow_log_path: result_eval_flow_raw/stage1_rpose/type/weights_15.log
            # if os.path.dirname(eval_flow_log_path):
            if not os.path.exists(os.path.dirname(opts.eval_flow_log_path.replace('type', 'all'))):
                os.makedirs(os.path.dirname(opts.eval_flow_log_path.replace('type', 'all')))
            if not os.path.exists(os.path.dirname(opts.eval_flow_log_path.replace('type', 'fg'))):
                os.makedirs(os.path.dirname(opts.eval_flow_log_path.replace('type', 'fg')))
            if not os.path.exists(os.path.dirname(opts.eval_flow_log_path.replace('type', 'bg'))):
                os.makedirs(os.path.dirname(opts.eval_flow_log_path.replace('type', 'bg')))

            with open(opts.eval_flow_log_path.replace('type', 'all'), 'w') as f:
                f.write('Mean Noc EPE = %.4f \n' % all_noc_mean_epe)
                f.write('Mean Occ EPE = %.4f \n' % all_occ_mean_epe)
            with open(opts.eval_flow_log_path.replace('type', 'fg'), 'w') as f:
                f.write('Mean Noc EPE = %.4f \n' % fg_noc_mean_epe)
                f.write('Mean Occ EPE = %.4f \n' % fg_occ_mean_epe)
            with open(opts.eval_flow_log_path.replace('type', 'bg'), 'w') as f:
                f.write('Mean Noc EPE = %.4f \n' % bg_noc_mean_epe)
                f.write('Mean Occ EPE = %.4f \n' % bg_occ_mean_epe)
            
