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

    def predict_disp_img_pair(self, inputs):
        if self.opt.SIG:
            disp_input_1 = torch.cat([inputs["tensorImg1"], inputs["img_1_sem"], inputs["img_1_edge"]], 1)
            disp_input_2 = torch.cat([inputs["tensorImg2"], inputs["img_2_sem"], inputs["img_2_edge"]], 1)
            disp_1 = self.predict_disp(disp_input_1)
            disp_2 = self.predict_disp(disp_input_2)
        else:
            disp_1 = self.predict_disp(inputs["tensorImg1"])
            disp_2 = self.predict_disp(inputs["tensorImg2"])
        return disp_1, disp_2
        

    def predict_rigid_flow(self, cam_T_cam, depth, inv_K_dict, K_dict):
        source_scale = 0
        inv_K = inv_K_dict[source_scale]
        K = K_dict[source_scale]
        outputs = dict()

        T = cam_T_cam # T from 0 to -1 or 0 to 1
        # cam_points of frame 0, [12, 4, 122880]
        cam_points = self.backproject_depth[source_scale](depth, inv_K)
        pix_coords = self.project_3d[source_scale](cam_points, K, T)
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
        outputs["flow"] = flow_pred

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
                    x2 = x2 + 20 if x2 <= (self.opt.width - 20) else (self.opt.width)
                    y2 = y2 + 20 if y2 <= (self.opt.height - 20) else (self.opt.height)

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
                x1, y1, x2, y2 = 0, 0, 640, 192

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

    def compute_IOU(self, mask1, mask2):
        """
        mask1: b, 1, h, w
        """
        inter = mask1 * mask2 # b,
        outer = 1 - (1-mask1) * (1-mask2) # b,
        IOU = inter.sum([2, 3]) * 1.0 / (outer.sum([2, 3])+1e-3) # b,
        return IOU
        
    def predict_rigid_flow_with_ins_flow(self, inputs, cam_T_cam, depth0,
                                        inv_K_dict, K_dict, rigid_flow):
        # some definitions
        scale = 0
        inv_K = inv_K_dict[scale]
        K = K_dict[scale]
        img0 = inputs["tensorImg1"]
        img1 = inputs["tensorImg2"]
        outputs = dict()
        
        # define the final image and mask
        img0_pred_final = torch.zeros([self.opt.batch_size, 3, self.opt.height, self.opt.width]).cuda()   # final image
        mask0_pred_final = torch.zeros([self.opt.batch_size, 1, self.opt.height, self.opt.width]).cuda() # bs, 1, 192, 640
        mask1_final = torch.zeros([self.opt.batch_size, 1, self.opt.height, self.opt.width]).cuda() # bs, 1, 192, 640
        
        # step2: compute pix_coords of img0_pred and flow
        T_static = cam_T_cam
        cam_points = self.backproject_depth[scale](
            depth0, inv_K) # cam_points of frame 0, [12, 4, 122880]
        pix_coords = self.project_3d[scale](
            cam_points, K, T_static)
        cam_coords0_pred = torch.matmul(T_static[:, :3, :], cam_points) # bs, 3, 122880
        ones = nn.Parameter(torch.ones(self.opt.batch_size, 1, self.opt.height * self.opt.width),
                             requires_grad=False).to(self.device)
        cam_coords0_pred = torch.cat([cam_coords0_pred, ones], 1) # bs, 4, 122880

        img0_pred = F.grid_sample(img1, pix_coords, padding_mode="border")
        flow_pred = self.extract_flow(pix_coords)
        flow_pred_final = torch.zeros_like(flow_pred)

        # FIXME: could delete this part
        # img0_pred_final = F.grid_sample(img1, pix_coords, padding_mode="border")
        # mask0_pred_final = F.grid_sample(torch.sum(inputs["img_2_ins"][:, 1:, :, :], 1).unsqueeze(1).float(), pix_coords, padding_mode="border")
        # flow_pred_final = flow_pred

        # step3: compute image feature and crop ROI feature
        img0_feature = self.models["instance_pose_encoder"](img0)[-1] # [bs, 512, 6, 20]
        img0_pred_feature = self.models["instance_pose_encoder"](img0_pred)[-1] # [bs, 512, 6, 20]
        
        if self.opt.use_depth_ordering:
            depth0_pred_final = 80.0 * torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width], requires_grad=True).to(self.device) # bs, 1, 192, 640
            
        instance_K_num = inputs["img_2_ins"].shape[1] - 1
        for ins_id in range(instance_K_num-1, -1, -1): # benefit to large instance            # step4: use T_static to transform mask of each ins
            # step4: use T_static to transform mask of each ins
            img1_ins_mask = inputs["img_2_ins"][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
            img0_pred_ins_mask = F.grid_sample(img1_ins_mask, pix_coords) #[b, 1, h, w]

            # step5: crop ins feature of img0 and img0_pred
            img0_pred_ins_bbox = self.extract_bbox_from_mask(img0_pred_ins_mask)
            # img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(3,3)) # [b, 512, 3, 3] 
            img0_pred_ins_feature = torchvision.ops.roi_align(img0_pred_feature, img0_pred_ins_bbox, output_size=(6,20)) # [b, 512, 3, 3]

            if self.opt.use_insid_match:
                img0_ins_feature = torch.cat([img0_ins_feature_list[i*instance_K_num+ins_id, :, :, :].unsqueeze(0) for i in range(self.opt.batch_size)])
            else:
                # use warped bbox
                # img0_ins_feature = torchvision.ops.roi_align(img0_feature, img0_pred_ins_bbox, output_size=(3,3))
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
            
            # FIXME: delete
            # delta = ins_cam_T_cam[0, :, 3]
            # T_dynamic = torch.zeros_like(T_dynamic).cuda()
            # T_dynamic[0, :, :] = torch.eye(4)
            # T_dynamic[0, :, 3] = delta
            # T_dynamic[0, 0, 3] = 0 # tx, not ok
            # T_dynamic[0, 1, 3] = 0 # ty, ok
            # T_dynamic[0, 2, 3] = 0 # tz, not ok
            # print(T_dynamic)

            if self.opt.use_depth_ordering:
                ins_pix_coords, img_z = self.project_3d[scale](cam_coords0_pred, K, T_dynamic, return_z=True)
            else:
                ins_pix_coords = self.project_3d[scale](cam_coords0_pred, K, T_dynamic)

            #step8: predict frame 0 from frame 1 based on T_dynamic and T_static
            # img0_pred_new = F.grid_sample(img1, ins_pix_coords, padding_mode="border")
            img0_pred_new = F.grid_sample(img0_pred, ins_pix_coords, padding_mode="border")
            # img0_pred_ins_mask_new = F.grid_sample(img1_ins_mask, ins_pix_coords) # [bs, 1, 192, 640]
            img0_pred_ins_mask_new = F.grid_sample(img0_pred_ins_mask, ins_pix_coords) # [bs, 1, 192, 640]

            #step8.5: use IOU value to filter invalid points
            if self.opt.iou_thres is not None:
                img0_ins_mask = inputs["img_1_ins"][:, ins_id+1, :, :].unsqueeze(1).float()

                ins_IOU = self.compute_IOU(img0_ins_mask, img1_ins_mask) # [b, 1]
                IOU_mask = ins_IOU > self.opt.iou_thres # [b, 1]

                img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, -1) # [b, 1x192x640]
                img0_pred_ins_mask_new = img0_pred_ins_mask_new * IOU_mask.float() # [b, 1x192x640]
                img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

            #step8.6: use diff between t_pred and t_gt to eliminate relative static area
            if self.opt.roi_diff_thres is not None:
                roi_abs = torch.abs(img0_pred * img0_pred_ins_mask - img0 * img0_pred_ins_mask)
                
                # roi_abs: bs, 3, 192, 640
                roi_sum = torch.sum(roi_abs, dim=[1, 2, 3]) # bs,
                mask_sum = torch.sum(img0_pred_ins_mask, dim=[1, 2, 3]) # bs,
                roi_diff = roi_sum * 1.0 / (mask_sum+1e-3) # bs,

                roi_diff = roi_diff.unsqueeze(1) # [bs, 1]
                roi_mask = roi_diff > self.opt.roi_diff_thres # [bs, 1]

                img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, -1) # [b, 1x192x640]
                img0_pred_ins_mask_new = img0_pred_ins_mask_new * roi_mask.float() # [b, 1x192x640]
                img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

            #step9: predict image and coords
            # img0_pred_final:[bs, 3, 192, 640], img0_pred_ins_mask_new: [bs, 1, 192, 640], ins_pix_coords: [bs, 192, 640, 2]
            if self.opt.use_depth_ordering:
                # img_z: bs, 1, 192, 640
                ins_z = img_z * img0_pred_ins_mask_new
                ins_z_mean = torch.sum(ins_z, [1, 2, 3]) / (torch.sum(img0_pred_ins_mask_new, [1, 2, 3])+1e-3)
                depth0_pred_mean = torch.sum(depth0_pred_final*img0_pred_ins_mask_new, [1, 2, 3]) / (torch.sum(img0_pred_ins_mask_new, [1, 2, 3])+1e-3)
                insz_less_than_depth = (ins_z_mean<depth0_pred_mean).unsqueeze(1) # bs, 1

                img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, -1) # [b, 1x192x640]
                img0_pred_ins_mask_new = img0_pred_ins_mask_new * insz_less_than_depth.float() # [b, 1x192x640]
                img0_pred_ins_mask_new = img0_pred_ins_mask_new.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

                depth0_pred_final = torch.add(depth0_pred_final*(1-img0_pred_ins_mask_new), img_z*img0_pred_ins_mask_new)
            
            img0_pred_final = torch.add(img0_pred_final*(1-img0_pred_ins_mask_new), img0_pred_new*img0_pred_ins_mask_new)
            mask0_pred_final = torch.add(mask0_pred_final*(1-img0_pred_ins_mask_new), img0_pred_ins_mask_new)
            
            # TODO:
            ins_flow_pred = self.extract_flow(ins_pix_coords)
            flow_pred_final = torch.add(flow_pred_final*(1-img0_pred_ins_mask_new.permute(0, 2, 3, 1)),
                ins_flow_pred*img0_pred_ins_mask_new.permute(0, 2, 3, 1))
            # flow_pred = torch.add(flow_pred*(1-img0_pred_ins_mask_new.permute(0, 2, 3, 1)),
            #     ins_flow_pred*img0_pred_ins_mask_new.permute(0, 2, 3, 1))
        
        # mask0_pred_final = torch.sum(inputs["img_1_ins"][:, 1:, :, :], 1).unsqueeze(1).float()
        color_ori = img0_pred
        color_new = mask0_pred_final * img0_pred_final + (1-mask0_pred_final) * color_ori


        outputs["0_tgt_{t}"] = inputs["tensorImg1"]
        outputs["1_src_{t+1}"] = inputs["tensorImg2"]
        outputs["2_pred_by_mono2"] = color_ori
        outputs["3_pred_by_(mono2+ins)"] = color_new
        outputs["4_flow_by_mono2"] = flow_pred
        outputs["5_flow_by_(mono2+ins)"] = mask0_pred_final.permute(0, 2, 3, 1) * flow_pred_final + (1-mask0_pred_final).permute(0, 2, 3, 1) * flow_pred
        # outputs["5_flow_by_(mono2+ins)"] = mask0.permute(0, 2, 3, 1) * flow_pred_final + (1-mask0).permute(0, 2, 3, 1) * flow_pred
        outputs["6_mask_pred_by_mono2"] = F.grid_sample(torch.sum(inputs["img_2_ins"][:, 1:, :, :], 1).unsqueeze(1).float(), pix_coords)
        outputs["7_mask_pred_by_(mono2+ins)"] = mask0_pred_final
        outputs["8_diff_mono2_(mono2+ins)"] = color_new - color_ori
        outputs["9_foreground"] = img0_pred_final # foreground
        outputs["10_gt_noc_flow"] = inputs["gt_noc_flow"]
        outputs["11_gt_occ_flow"] = inputs["gt_occ_flow"]
        # outputs["10_fg_flow"] = mask0_pred_final.permute(0, 2, 3, 1) * flow_pred_final
        # outputs["11_bg_flow"] = (1-mask0_pred_final).permute(0, 2, 3, 1) * flow_pred

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
                                 rigid_flow_outputs["flow"])
            else:
                flow_outputs = self.predict_rigid_flow_with_ins_flow(
                                            inputs, cam_T_cam, depth_1,
                                            self.inv_K, self.K,
                                            rigid_flow_outputs["flow"])
        else:
            #FIXME: whether use ins_pose_net
            # only compute rigid_flow
            flow_outputs = self.predict_rigid_flow(cam_T_cam, depth_1, self.inv_K, self.K)

        return flow_outputs

    def load_img(self, input_dir, input_file):
        input_path_1 = os.path.join(input_dir, input_file+"_10.png")
        input_path_2 = os.path.join(input_dir, input_file+"_11.png")

        #[h, w, 3] -> [3, h, w]->[1, 3, h, w]
        resize_func = transforms.Resize((self.opt.height, self.opt.width),
                                        interpolation=PIL.Image.ANTIALIAS)

        img1 = resize_func(PIL.Image.open(input_path_1).convert('RGB'))
        img2 = resize_func(PIL.Image.open(input_path_2).convert('RGB'))

        # [b, 3, h, w]
        tensorImg1 = transforms.ToTensor()(img1).unsqueeze(0).to(self.device)
        tensorImg2 = transforms.ToTensor()(img2).unsqueeze(0).to(self.device)

        # load gt flow 
        gt_noc_fn = os.path.join('dataset/data_scene_flow/training/flow_noc/'+input_file+"_10.png")
        gt_occ_fn = os.path.join('dataset/data_scene_flow/training/flow_occ/'+input_file+"_10.png")

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
        for key in flow_outputs:
            if "5_flow" not in key:
                continue

            optical_flow = flow_outputs[key]
            optical_flow = optical_flow[0].cpu().numpy()
            if os.path.exists(os.path.join(output_dir, "npy")) == False:
                os.makedirs(os.path.join(output_dir, "npy"))
            out_path = os.path.join(output_dir, "npy", input_file+".npy")
            np.save(out_path, optical_flow)
    
    def save_disp(self, disp1, disp2, output_dir, input_file):
        # if os.path.exists(os.path.join(output_dir, "disp")) == False:
        #     os.makedirs(os.path.join(output_dir, "disp"))

        out_path1 = os.path.join(output_dir, input_file+"_10.npy")
        np.save(out_path1, disp1.cpu().numpy())
        out_path2 = os.path.join(output_dir, input_file+"_11.npy")
        np.save(out_path2, disp2.cpu().numpy())

    def visualize_flow(self, flow_outputs, output_dir, input_file):
        # output_dir: output/flow/${MODEL}/weights_${EPOCH}
        # flow_outputs: color_ori, color_diff, color, f_img_syn, warped_mask, flow
        '''
        color_ori torch.Size([1, 3, 192, 640])
        color_diff torch.Size([1, 3, 192, 640])
        color torch.Size([1, 3, 192, 640])
        f_img_syn torch.Size([1, 3, 192, 640])
        warped_mask torch.Size([1, 1, 192, 640])
        flow torch.Size([1, 192, 640, 2])
        '''

        vis_output_dir = output_dir.replace('flow', 'vis')
        if os.path.exists(vis_output_dir) == False:
            os.makedirs(vis_output_dir)

        output_path = os.path.join(vis_output_dir, input_file+".jpg")

        plt.figure(figsize=(12, 15))

        with warnings.catch_warnings():
            for i, key in enumerate(flow_outputs):
                plt.subplot(6, 2, i+1)
                plt.title(key)
                cur_img = flow_outputs[key][0].cpu().numpy()

                if 'flow' in key:
                    cur_img = fl.flow_to_image(cur_img)
                elif "mask_pred" in key:
                    cur_img = np.transpose(cur_img, (1, 2, 0))[:,:,0]
                else:
                    cur_img = np.transpose(cur_img, (1, 2, 0))
                '''
                if 'diff' in key:
                    cur_img = (cur_img + 1) / 2.
                '''
                plt.imshow(cur_img)
                # plt.xticks([])
                # plt.yticks([])

        #plt.axis('off')
        plt.savefig(output_path)
        plt.close()
    
    def prepare_inputs(self, tensorImg1, tensorImg2, gt_noc_flow, gt_occ_flow,
                        img_1_sem, img_2_sem, img_1_bbox, img_1_ins, 
                        img_1_edge, img_2_bbox, img_2_ins, img_2_edge):
        inputs = {}
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
            inputs[key] = ipt.to(self.device)
        return inputs

if __name__ == '__main__':
    # load Model
    options = MonodepthOptions()
    opts = options.parse()
    evaler = Evaler(opts)

    input_dir = opts.input_dir
    output_dir = opts.output_dir # output/flow/${MODEL}/weights_${EPOCH}

    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)

    save_flag = True

    bbox_dir = "dataset/data_scene_flow_SIG/bbox"
    sem_dir = "dataset/data_scene_flow_SIG/sem"
    ins_dir = "dataset/data_scene_flow_SIG/ins"

    # selected_idx = [136,91,133,43,0,182,29,6,63,68]
    selected_idx = [20, 43, 44, 45, 46, 49, 58, 91, 95, 184, 197]
    # selected_idx = [184] 
    with torch.no_grad():
        for i in tqdm(range(200)):
        # for i in selected_idx:
            # load img pair
            input_file = str(i).zfill(6)

            tensorImg1, tensorImg2, gt_noc_flow, gt_occ_flow = evaler.load_img(input_dir, input_file)
            img_1_sem, img_2_sem = evaler.load_sem(sem_dir, input_file)
            img_1_bbox, img_1_ins, img_1_edge, img_2_bbox, img_2_ins, img_2_edge = \
                evaler.load_bbox_ins_edge(bbox_dir, ins_dir, input_file)

            inputs = evaler.prepare_inputs(tensorImg1, tensorImg2, gt_noc_flow, gt_occ_flow, 
                                img_1_sem, img_2_sem, img_1_bbox, img_1_ins, 
                                img_1_edge, img_2_bbox, img_2_ins, img_2_edge)
            
            disp1, disp2 = evaler.predict_disp_img_pair(inputs)

            # save flow
            if save_flag == True:
                evaler.save_disp(disp1, disp2, output_dir, input_file)

