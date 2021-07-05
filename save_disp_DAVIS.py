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
            disp_1 = self.predict_disp(disp_input_1)
        else:
            disp_1 = self.predict_disp(inputs["tensorImg1"])
        return disp_1
        

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
        
    def load_img(self, input_dir, input_file):
        input_path_1 = os.path.join(input_dir, input_file)

        #[h, w, 3] -> [3, h, w]->[1, 3, h, w]
        resize_func = transforms.Resize((self.opt.height, self.opt.width),
                                        interpolation=PIL.Image.ANTIALIAS)

        img1 = resize_func(PIL.Image.open(input_path_1).convert('RGB'))

        # [b, 3, h, w]
        tensorImg1 = transforms.ToTensor()(img1).unsqueeze(0).to(self.device)
        return tensorImg1

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
        input_path_1 = os.path.join(input_dir, input_file.replace("jpg", "npy"))

        sem_1 = Image.fromarray(np.uint8(self.get_sem_ins(input_path_1))).convert("L")

        sem_1 = sem_1.resize((self.opt.width, self.opt.height), Image.NEAREST)

        sem_1_onehot = self.convert_sem_onehot(sem_1)

        return sem_1_onehot

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

    def extract_bbox_ins_edge(self, bbox_dir, ins_dir, file_name):
        bbox_path = os.path.join(bbox_dir, file_name.replace("jpg", "txt"))
        ins_path = os.path.join(ins_dir, file_name.replace("jpg", "npy"))

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

    
    def load_bbox_ins_edge(self, bbox_dir, ins_dir, input_file):
        ins_RoI_bbox_1, ins_onehot_1, ins_edge_1 = self.extract_bbox_ins_edge(bbox_dir, ins_dir, input_file)

        return ins_RoI_bbox_1, ins_onehot_1, ins_edge_1

    def save_disp(self, disp1, output_dir, input_file):
        out_path1 = os.path.join(output_dir, input_file)
        np.save(out_path1, disp1.cpu().numpy())

    
    def prepare_inputs(self, tensorImg1):
        inputs = {}
        inputs["tensorImg1"] = tensorImg1
        '''
        inputs["img_1_bbox"] = img_1_bbox
        inputs["img_1_sem"] = img_1_sem
        inputs["img_1_ins"] = img_1_ins
        inputs["img_1_edge"] = img_1_edge
        '''
        
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

    # rgd_dir = "/home/xzwu/xzwu/Code/GenerateSemantic/Dataset/DAVIS_2017/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/JPEGImages/Full-Resolution"
    # rgd_dir = "/home/xzwu/xzwu/Code/GenerateSemantic/Dataset/DAVIS_2017/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/JPEGImages/Full-Resolution"
    # sem_dir = "/home/xzwu/xzwu/Code/GenerateSemantic/Dataset/DAVIS_2017/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/SIG/sem"
    # ins_dir = "/home/xzwu/xzwu/Code/GenerateSemantic/Dataset/DAVIS_2017/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/SIG/ins"
    # bbox_dir = "/home/xzwu/xzwu/Code/GenerateSemantic/Dataset/DAVIS_2017/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/SIG/bbox"
    # input_file_list = os.listdir(rgd_dir)
    img_ext = "jpg"

    # total_img = 200
    class_list = os.listdir(input_dir)
    with torch.no_grad():
        for cur_class in class_list:
            if 'sheep' in cur_class:
                continue
            print(cur_class)
            sub_dir = os.path.join(input_dir, cur_class)
            input_file_list = os.listdir(sub_dir)
            total_img = len(input_file_list)
            for i in tqdm(range(total_img-1)):
                try:
                    # load img pair
                    input_file = str(i).zfill(5)+"."+img_ext
                    tensorImg1 = evaler.load_img(os.path.join(input_dir, cur_class), input_file)
                    inputs = evaler.prepare_inputs(tensorImg1)
                    disp1 = evaler.predict_disp_img_pair(inputs)

                    # save flow
                    if save_flag == True:
                        output_path = os.path.join(output_dir, cur_class)
                        if os.path.exists(output_path) == False:
                            os.mkdir(output_path)
                        evaler.save_disp(disp1, output_path, input_file.replace('jpg', 'npy'))
                
                except Exception as e:
                    print(e)
                    continue