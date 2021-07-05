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
import argparse

import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, './flow_tool/')
import flowlib as fl

class BackprojectDepth(nn.Module): 
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T, return_z=False):
        P = torch.matmul(K, T)[:, :3, :]
        cam_points = torch.matmul(P, points) # bs, 3, 12288
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2

        if return_z == False:
            return pix_coords
        else:
            return pix_coords, cam_points[:, 2, :].unsqueeze(1).view(self.batch_size, 1, self.height, self.width)


class Transform3D(nn.Module):
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Transform3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

    # def forward(self, points, K, T, return_z=False):
    def forward(self, points, T, return_z=False):
        # P = torch.matmul(K, T)[:, :3, :]
        P = T[:, :3, :]
        cam_points = torch.matmul(P, points) # bs, 3, 12288
        cam_points = torch.cat([cam_points, self.ones], 1) # bs, 4, 12288
        
        # return cam_points.view(self.batch_size, 1, self.height, self.width)
        return cam_points

class Evaler:
    def __init__(self, options):
        self.opt = options

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.device = torch.device("cuda")

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

        print("Eval is using:\n  ", self.device)
        self.backproject_depth = {}
        self.project_3d = {}
        self.transform_3d = {}
        for scale in range(4): # [0,1,2,3]
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            # initialize backproject_depth and project_3d at each scale
            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)
            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)
            self.transform_3d[scale] = Transform3D(self.opt.batch_size, h, w)
            self.transform_3d[scale].to(self.device)

    # def predict_depth(self, input_tensor):
    #     #step0: compute disp
    #     features = self.models["encoder"](input_tensor)
    #     outputs = self.models["depth"](features)
    #     disp = outputs[("disp", 0)]
    #     disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

    #     #step1: compute depth
    #     # min_depth = 0.1, max_depth = 100
    #     _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

    #     return depth
    
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
            img0_pred = F.grid_sample(img1, pix_coords,padding_mode="border")

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
        
        index2_x_delta = first_flow_x.to(dtype=torch.int64)
        index2_y_delta = first_flow_y.to(dtype=torch.int64)

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

    def match_cam_points(self, cam_points_2, optical_flow_before):#, height, width):
        """ 
        reorder cam_points_2 to match cam_points_1
        """
        # print(cam_points_1.shape, cam_points_2.shape) # [1, 4, 122880]
        width = self.opt.width
        height = self.opt.height
        
        # cam_points_1_x = cam_points_1[0, 0, :] # [192*640,]
        # cam_points_1_y = cam_points_1[0, 1, :] # [192*640,]
        # cam_points_1_z = cam_points_1[0, 2, :] # [192*640,]

        cam_points_2_x = cam_points_2[0, 0, :] # [192*640,]
        cam_points_2_y = cam_points_2[0, 1, :] # [192*640,]
        cam_points_2_z = cam_points_2[0, 2, :] # [192*640,]

        # print(optical_flow.shape) # [1, 375, 1242, 3] 
        optical_flow_before = optical_flow_before[0, :, :, :].cpu().numpy()
        optical_flow = fl.resize_flow(optical_flow_before, width, height)
        optical_flow = torch.Tensor(optical_flow).cuda()
        
        # print(optical_flow.shape) # (192, 640, 3)
        last_channel = optical_flow.shape[2]
        optical_flow = optical_flow.view(-1, last_channel)
        optical_flow_x = optical_flow[:, 0] # [192*640,]
        optical_flow_y = optical_flow[:, 1] # [192*640,]
        index2_x_delta = optical_flow_x.to(dtype=torch.int64)
        index2_y_delta = optical_flow_y.to(dtype=torch.int64)
        
        ori_yy, ori_xx = torch.meshgrid(torch.arange(0, height), torch.arange(0, width))
        ori_xx = torch.reshape(ori_xx, (-1, 1)).to(self.device)
        ori_yy = torch.reshape(ori_yy, (-1, 1)).to(self.device)
        index2_x = torch.clamp(ori_xx.squeeze() + index2_x_delta, 0, width-1)
        index2_y = torch.clamp(ori_yy.squeeze() + index2_y_delta, 0, height-1)
        index2 = index2_y * width + index2_x
        
        # print(cam_points_2_x.shape) # [192*640,]
        cam_points_2_after_x = torch.gather(cam_points_2_x, 0, index2) #.to(dtype=torch.int64)
        cam_points_2_after_y = torch.gather(cam_points_2_y, 0, index2)
        cam_points_2_after_z = torch.gather(cam_points_2_z, 0, index2)

        # print(cam_points_2_after_x.shape) # [192*640,]
        cam_points_2_after = torch.cat([cam_points_2_after_x.unsqueeze(0), cam_points_2_after_y.unsqueeze(0), cam_points_2_after_z.unsqueeze(0)], dim=0).unsqueeze(0)
        # print(cam_points_2_after.shape) # [1, 3, 192*640]
        ones = nn.Parameter(torch.ones(1, 1, height * width), requires_grad=False).cuda()
        cam_points_2_after = torch.cat([cam_points_2_after, ones], 1) # [1, 4, 192*640]
        
        '''
        # print(optical_flow.shape) # [1, 375, 1242, 3]
        optical_flow = fl.resize_flow(optical_flow_before[0, :, :, :].cpu().numpy(), width, height)
        # print(optical_flow.shape) # (192, 640, 3)
        optical_flow = optical_flow.reshape(-1, 3)

        optical_flow_x = optical_flow[:, 0] # [192*640,]
        optical_flow_y = optical_flow[:, 1] # [192*640,]
        index2_x_delta = optical_flow_x.astype(np.int64)
        index2_y_delta = optical_flow_y.astype(np.int64)
        
        ori_yy, ori_xx = np.meshgrid(np.arange(0, height), np.arange(0, width))
        ori_xx = np.reshape(ori_xx, (-1, 1))
        ori_yy = np.reshape(ori_yy, (-1, 1))
        index2_x = np.clip(ori_xx.squeeze() + index2_x_delta, 0, width-1)
        index2_y = np.clip(ori_yy.squeeze() + index2_y_delta, 0, height-1)
        index2 = torch.Tensor(index2_y * width + index2_x).cuda()

        cam_points_2_after_x = torch.gather(torch.Tensor(cam_points_2_x).cuda(), 0, index2) #.to(dtype=torch.int64)
        cam_points_2_after_y = torch.gather(torch.Tensor(cam_points_2_y).cuda(), 0, index2)
        cam_points_2_after_z = torch.gather(torch.Tensor(cam_points_2_z).cuda(), 0, index2)

        cam_points_2_after = torch.cat([cam_points_2_after_x.unsqueeze(0), cam_points_2_after_y.unsqueeze(0), cam_points_2_after_z.unsqueeze(0)], dim=0).unsqueeze(0)
        ones = nn.Parameter(torch.ones(1, 1, height * width), requires_grad=False).cuda()
        cam_points_2_after = torch.cat([cam_points_2_after, ones], 1) # [1, 4, 192*640]
        '''

        return cam_points_2_after, optical_flow

    def predict_3d_flow_src(self, cam_T_cam, depth_1, depth_2, optical_flow,
                        inv_K_dict, K_dict, img1=None):
        """ compute 3d scene flow gt in src system
        """
        source_scale = 0
        inv_K = inv_K_dict[source_scale]
        K = K_dict[source_scale]
        outputs = dict()

        T = cam_T_cam # T from 0 to -1 or 0 to 1
        # cam_points of frame 0, [1, 4, x]
        cam_points_1 = self.backproject_depth[source_scale](depth_1, inv_K)
        cam_points_1_after = self.transform_3d[source_scale](cam_points_1, T) # [1, 3, x]
        cam_points_2 = self.backproject_depth[source_scale](depth_2, inv_K) # [1, 4, 122880]
        cam_points_2_after = self.match_cam_points(cam_points_2, optical_flow)
        
        flow_pred = cam_points_2_after - cam_points_1_after
        # flow_pred = cam_points_2 - cam_points_1_after
        # print(flow_pred.shape) # [1, 4, x]
        flow_pred = flow_pred.view(self.opt.batch_size, -1, self.opt.height, self.opt.width)
        flow_pred = flow_pred[:, :3, :, :]

        # if img1 is not None:
        #     # img
        #     outputs["img_stage1"] = img0_pred
        
        # flow
        outputs["flow_stage1"] = flow_pred

        return outputs

    def inv_transform_matrix(self, T):
        inv_T2 = T.clone()
        inv_T2[:, :3,3:] = -T[:, :3,3:]
        inv_T2[:, :3,:3] = torch.inverse(T[:, :3,:3])
        # print(inv_T2)
        
        inv_T = T.clone()
        inv_T[:, :3,3:] = torch.bmm(torch.inverse(T[:, :3,:3]), -T[:, :3,3:])
        # 0 = np.matmul(global_pose[:3,:3], rel_pose[:3,3:]) + global_pose[:3,3:]
        # E = np.matmul(global_pose[:3,:3], rel_pose[:3,:3])
            
        inv_T[:, :3,:3] = torch.inverse(T[:, :3,:3])
        # print(inv_T)
        
        # input()
        return inv_T
        
    def predict_3d_flow_tgt(self, cam_T_cam, depth_1, depth_2, resflow,
                        inv_K_dict, K_dict, img1=None):
        """ compute 3d scene flow gt in tgt system
        """
        source_scale = 0
        inv_K = inv_K_dict[source_scale]
        K = K_dict[source_scale]
        outputs = dict()

        T = cam_T_cam # T from 0 to -1 or 0 to 1
        # cam_points of frame 0, [1, 4, x] 
        cam_points_1 = self.backproject_depth[source_scale](depth_1, inv_K) # tgt
        # cam_points_1_after = self.transform_3d[source_scale](cam_points_1, T) # [1, 3, x]
        cam_points_2 = self.backproject_depth[source_scale](depth_2, inv_K) # [1, 4, 122880]
        
        rigid_flow_outputs = self.predict_rigid_flow(cam_T_cam, depth_1, self.inv_K, self.K)
        rigid_flow = rigid_flow_outputs['flow_stage1']
        # print(rigid_flow.shape, resflow.shape)
        # input()
        optical_flow = rigid_flow + resflow[:,:,:,:2]
        # optical_flow = torch.unsqueeze(self.add_flow(rigid_flow[0], resflow[0,:,:,:2]), 0)
        # optical_flow = torch.unsqueeze(self.add_flow(resflow[0,:,:,:2], rigid_flow[0]), 0)
        

        inv_T = torch.inverse(T) #self.inv_transform_matrix(T)
        cam_points_2_after = self.transform_3d[source_scale](cam_points_2, inv_T) # tgt
        cam_points_2_after, optical_flow = self.match_cam_points(cam_points_2_after, optical_flow)
        
        flow_pred = cam_points_2_after - cam_points_1
        flow_pred = flow_pred.view(self.opt.batch_size, -1, self.opt.height, self.opt.width)
        flow_pred = flow_pred[:, :3, :, :]
        
        # flow
        '''
        mask = optical_flow.view(self.opt.height, self.opt.width, -1)[:, :, 2]
        # mask = torch.stack([mask, mask, mask], 0).unsqueeze(0) # 1, 3, 192, 640
        mask = mask.unsqueeze(0).unsqueeze(0) # 1, 1, 192, 640
        
        # outputs["flow_stage1"] = flow_pred * mask
        flow_pred = torch.cat([flow_pred, mask], 1)
        '''
        outputs["flow_stage1"] = flow_pred
        return outputs

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

    def predict_flow_img_pair(self, inputs):
        img0 = inputs["tensorImg1"]
        img1 = inputs["tensorImg2"]

        depth0 = inputs["depth1"]
        depth1 = inputs["depth2"]
        cam_T_cam = inputs["pose1"]
        # optical_flow = inputs["gt_noc_flow"]
        optical_flow = inputs["resflow"]

        flow_outputs = self.predict_3d_flow_tgt(cam_T_cam, depth0, depth1, optical_flow, self.inv_K, self.K, img1)

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

    def load_depth(self, input_dir, input_file, img_ext='npz', idx=None):
        input_path_1 = os.path.join(input_dir, 'depth', input_file+"_10."+img_ext)
        input_path_2 = os.path.join(input_dir, 'depth', input_file+"_11."+img_ext)

        depth1 = np.load(input_path_1)["depth"]
        depth2 = np.load(input_path_2)["depth"]

        return depth1, depth2
    
    def load_pose(self, input_dir, input_file, img_ext='npy'):
        input_path_1 = os.path.join(input_dir, input_file+"_10."+img_ext)
        pose = np.load(input_path_1)
        return pose

    def load_resflow(self, input_dir, input_file, img_ext='png'):
        input_path_1 = os.path.join(input_dir, input_file+"."+img_ext)
        resflow = fl.read_flow(input_path_1)
        return resflow

    def save_flow(self, flow_outputs, output_dir, input_file):
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

    def visualize_3d_flow(self, flow_outputs, output_dir, input_file, ins_dir, inputs=None):
        # output_dir: output/flow/${MODEL}/weights_${EPOCH}
        if self.opt.dataset == 'drivingstereo_eigen':
            vis_output_dir = output_dir.replace('flow_ds', 'flow').replace('flow', 'flow_ds').replace('flow', 'vis')
        elif self.opt.dataset == 'kitti_raw_data_flow':
            vis_output_dir = output_dir.replace('flow_raw', 'flow').replace('flow', 'flow_raw').replace('flow', 'vis')
        else:
            vis_output_dir = output_dir.replace('flow', 'vis')
        
        # TODO: add
        vis_output_dir = os.path.join(vis_output_dir, 'vis')
        if os.path.exists(vis_output_dir) == False:
            os.makedirs(vis_output_dir)

        output_path = os.path.join(vis_output_dir, input_file+".jpg")
        plt.figure(figsize=(15, 16))

        for i, key in enumerate(flow_outputs):
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
                elif 'mask' in key:
                    cur_img = np.transpose(cur_img, (1, 2, 0))[:,:,0]
                elif 'flow_tgt' in key:
                    cur_img = fl.resize_flow(cur_img, self.opt.width, self.opt.height)
                    cur_img = fl.flow_to_image(cur_img)
                elif 'flow' in key:
                    # evaluate flow, mono2 and ours
                    # pred_flow = flow_outputs[key]
                    # if self.opt.dataset == 'drivingstereo_eigen':
                    #     pass
                    # else:
                    #     pass
                    #     # fg_single_noc_epe, fg_single_occ_epe = self.compute_flow_EPE(inputs["gt_noc_flow"][0].cpu().numpy(), inputs["gt_occ_flow"][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'fg', ins_dir, input_file)
                    #     # all_single_noc_epe, all_single_occ_epe = self.compute_flow_EPE(inputs["gt_noc_flow"][0].cpu().numpy(), inputs["gt_occ_flow"][0].cpu().numpy(), pred_flow[0].cpu().numpy(), 'all', ins_dir, input_file)
                    #     # key = '%s: fg: %.2f, %.2f, all: %.2f,%.2f' % (key, fg_single_noc_epe, fg_single_occ_epe, all_single_noc_epe, all_single_occ_epe)
                    
                    # cur_img = fl.flow_to_image(cur_img)
                    cur_img = (np.transpose(cur_img, (1,2,0)) + 127.0)/255.0
                    print(np.mean(cur_img), np.max(cur_img), np.min(cur_img))

                plt.imshow(cur_img)

            plt.title(key)
            # plt.colorbar()

        #plt.axis('off')
        plt.savefig(output_path)
        plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Compute Flow')
    # parser.add_argument('--model_input', type=str, default='output/PackNet01_MR_selfsup_K', help='Model input')
    parser.add_argument('--output_dir', type=str, default='output', help='Output file or folder')
    parser.add_argument('--dataset', type=str, choices=['kitti', 'kitti_raw_data_flow', 'drivingstereo_eigen'], default='kitti',
                        help='Dataset')
    parser.add_argument('--height', type=int, default=192,
                        help='image height')
    parser.add_argument('--width', type=int, default=640,
                        help='image width')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
    parser.add_argument('--depth_npy', type=str, default=None, help='depth input path')
    parser.add_argument('--depth_dir', type=str, default=None) #'/home/xzwu/xzwu/Dataset/Kitti/data_scene_flow/training/disp_occ_1'
    parser.add_argument('--pose', type=str, default='log/mono_fm_from_scratch_new/pose_pred_sceneflow.npy', help='pose input path')
    

    args = parser.parse_args()
    return args

def main(args):
    bbox_dir = "dataset/data_scene_flow_SIG/bbox"
    sem_dir = "dataset/data_scene_flow_SIG/sem"
    ins_dir = "dataset/data_scene_flow_SIG/ins"
    input_dir = "dataset/data_scene_flow/training/image_2"
    img_ext = "png"
    depth_dir = "/home/xzwu/xzwu/Code/MonoDepth2_stage1/GeoNet/depth"
    pose_dir = "/home/xzwu/xzwu/Code/MonoDepth2_stage1/GeoNet/pose"
    resflow_dir = "/home/xzwu/xzwu/Code/MonoDepth2_stage1/GeoNet/png"
    
    save_flag = True

    # load Model
    evaler = Evaler(args)
    output_dir = args.output_dir # output/flow/${MODEL}/weights_${EPOCH}
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    with torch.no_grad():
        # load img pair
        # t1 = timeit.default_timer()
        i = 20
        input_file = str(i).zfill(6)

        inputs = {}
        tensorImg1, tensorImg2, gt_noc_flow, gt_occ_flow = evaler.load_img(input_dir, input_file, img_ext, idx=i)
        pose = evaler.load_pose(pose_dir, input_file, img_ext='npy')
        rel_pose = np.expand_dims(pose, 0) # evaler.load_pose(FeatDepth_dir, input_file)
        resflow = evaler.load_resflow(resflow_dir, input_file)
        resflow = np.expand_dims(resflow, 0)
        
        inputs["tensorImg1"] = tensorImg1
        inputs["tensorImg2"] = tensorImg2
        
        inputs["pose1"] = torch.Tensor(rel_pose)
        inputs["resflow"] = torch.Tensor(resflow)
        inputs["gt_noc_flow"] = torch.Tensor(gt_noc_flow).unsqueeze(0)
        inputs["gt_occ_flow"] = torch.Tensor(gt_occ_flow).unsqueeze(0)
        
        depth1 = np.load(os.path.join(depth_dir, input_file+'_10.npy'))
        depth1 = cv2.resize(depth1, (args.width, args.height))
        inputs["depth1"] = torch.Tensor(depth1)
        depth2 = np.load(os.path.join(depth_dir, input_file+'_11.npy'))
        depth2 = cv2.resize(depth2, (args.width, args.height))
        inputs["depth2"] = torch.Tensor(depth2)

        for key, ipt in inputs.items():
            inputs[key] = ipt.to("cuda")
        
        # compute flow in MonoDepth2
        flow_outputs = evaler.predict_flow_img_pair(inputs)

        # save flow
        if save_flag == True:
            evaler.save_flow(flow_outputs, output_dir, input_file)

        # visualization
        # if vis_flag == True:
        #     evaler.visualize_3d_flow(flow_outputs, output_dir, input_file, ins_dir, inputs)

if __name__ == '__main__':
    args = parse_args()
    main(args)
