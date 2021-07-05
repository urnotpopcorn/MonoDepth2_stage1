# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks

class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        if self.opt.SIG:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained", mode = "SIG", cur="depth")
        else:
            self.models["encoder"] = networks.ResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")

        self.models["encoder"].to(self.device)
        
        if not self.opt.fix_depth:
            self.parameters_to_train += list(self.models["encoder"].parameters())
        else:
            for p in self.models["encoder"].parameters():
                p.requires_grad = False
            self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)

        if not self.opt.fix_depth:
            self.parameters_to_train += list(self.models["depth"].parameters())
        else:
            for p in self.models["depth"].parameters():
                p.requires_grad = False
            self.parameters_to_train += list(self.models["depth"].parameters())

        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                # if self.opt.SIG:
                #     self.models["pose_encoder"] = networks.ResnetEncoder(
                #         self.opt.num_layers,
                #         self.opt.weights_init == "pretrained",
                #         num_input_images=self.num_pose_frames*3, mode="SIG", cur="pose")
                # else:
                #     self.models["pose_encoder"] = networks.ResnetEncoder(
                #         self.opt.num_layers,
                #         self.opt.weights_init == "pretrained",
                #         num_input_images=self.num_pose_frames*3)
                self.models["pose_encoder"] = networks.ResnetPoseEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)

                self.models["pose_encoder"].to(self.device)

                if not self.opt.fix_pose:
                    self.parameters_to_train += list(self.models["pose_encoder"].parameters())
                else:
                    pose_encoder_para = self.models["pose_encoder"].parameters()
                    for p in pose_encoder_para:
                        p.requires_grad = False
                    self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                    #self.parameters_to_train += list(pose_encoder_para)

                # self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            elif self.opt.pose_model_type == "shared":
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, self.num_pose_frames)

            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)

            self.models["pose"].to(self.device)

            if not self.opt.fix_pose:
                self.parameters_to_train += list(self.models["pose"].parameters())
            else:
                pose_decoder_para = self.models["pose"].parameters()
                for p in pose_decoder_para:
                    p.requires_grad = False
                self.parameters_to_train += list(self.models["pose"].parameters())

                #self.parameters_to_train += list(pose_decoder_para)

            # self.parameters_to_train += list(self.models["pose"].parameters())

        # --------------------------------------stage2------------------------------------------
        if self.opt.instance_pose:
            def weight_init(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_normal_(m.weight)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, torch.nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)

            # define instance pose encoder
            # self.models["instance_pose_encoder"] = networks.ResnetPoseEncoder(
            #     self.opt.num_layers,
            #     self.opt.weights_init == "pretrained",
            #     num_input_images=self.num_pose_frames)
            self.models["instance_pose_encoder"] = networks.InsResnetEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained")
            self.models["instance_pose_encoder"].to(self.device)

            if not self.opt.fix_ins_pose:
                self.parameters_to_train += list(self.models["instance_pose_encoder"].parameters())
            else:
                ins_pose_encoder_para = self.models["instance_pose_encoder"].parameters()
                for p in ins_pose_encoder_para:
                    p.requires_grad = False
                self.parameters_to_train += list(self.models["instance_pose_encoder"].parameters())

            # define instance pose decoder
            # self.models["instance_pose"] = networks.PoseDecoder(
            #     self.models["instance_pose_encoder"].num_ch_enc,
            #     num_input_features=1,
            #     num_frames_to_predict_for=2)
            self.models["instance_pose"] = networks.InsPoseDecoder(
                    num_RoI_cat_features=1024,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            # FIXME: whether to init
            # self.models["instance_pose_encoder"].apply(weight_init)
            self.models["instance_pose"].apply(weight_init)
            self.models["instance_pose"].to(self.device)
            if not self.opt.fix_ins_pose:
                self.parameters_to_train += list(self.models["instance_pose"].parameters())
            else:
                ins_pose_para = self.models["instance_pose"].parameters()
                for p in ins_pose_para:
                    p.requires_grad = False
                self.parameters_to_train += list(self.models["instance_pose"].parameters())

            # --------------------------------------stage3------------------------------------------
            if self.opt.instance_motion:
                # stage3 is based on stage2
                self.models["instance_motion"] = networks.InsMotionDecoder(
                    self.models["instance_pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)
                
                if self.opt.instance_motion_weight_init:
                    self.models["instance_motion"].apply(weight_init)
                self.models["instance_motion"].to(self.device)

                if not self.opt.fix_ins_motion:
                    self.parameters_to_train += list(self.models["instance_motion"].parameters())
                else:
                    for p in self.models["instance_motion"].parameters():
                        p.requires_grad = False
                    self.parameters_to_train += list(self.models["instance_motion"].parameters())

        if self.opt.predictive_mask:
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        # --------------------------------------------------------------------------------
        # 1e-4 -> 1e-5 (15 epoch) -> 1e-6 (30 epoch), step=15
        # self.model_optimizer = optim.Adam(
            # filter(lambda p: p.requires_grad, self.parameters_to_train), self.opt.learning_rate)
        if self.opt.add_l2reg:
            self.model_optimizer = optim.AdamW(self.parameters_to_train, self.opt.learning_rate)
        else:
            self.model_optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters_to_train), self.opt.learning_rate)
            # self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)

        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "drivingstereo_eigen": datasets.DSRAWDataset}

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        '''
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opt=self.opt, mode="train")
        '''
        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opt=self.opt, mode="train")
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        '''
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opt=self.opt, mode="val")
        '''
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opt=self.opt, mode="val")
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 #and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                if self.opt.instance_pose:
                    if self.opt.mask_loss_weight is not None:
                        self.log_time(batch_idx, duration, losses["loss"].cpu().data,
                            losses["ins_loss"].cpu().data, losses["bg_loss"].cpu().data,
                            losses["mask_loss"].cpu().data)
                    else:
                        self.log_time(batch_idx, duration, losses["loss"].cpu().data,
                            losses["ins_loss"].cpu().data, losses["bg_loss"].cpu().data)
                else:
                    self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if self.opt.instance_pose:
                    # # print T_dynamic
                    # for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                    #     for ins_id in range(4):
                    #         print(outputs[("T_dynamic", frame_id, ins_id)])
                    print(outputs[("T_dynamic", 1, 0)][0])
                    # print(outputs[("delta_x_mean", 1, 0)])
                    pass

                '''
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                '''
                # self.log("train", inputs, outputs, losses)
                '''
                self.val()
                '''
            #input()
            self.step += 1
    
    def add_mask_losses(self, outputs, losses):
        scale = 0
        cur_mask = outputs[("cur_mask", 0, scale)]
        mse_loss = nn.MSELoss()
        mask_loss = 0

        for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
            warped_mask = outputs[("warped_mask", frame_id, scale)]
            # mask_loss += mse_loss(cur_mask, warped_mask)
            mask_loss += (1 - self.compute_IOU(cur_mask, warped_mask).mean())

        mask_loss /= len(self.opt.frame_ids[1:])
        mask_loss *= self.opt.mask_loss_weight
        losses['mask_loss'] = mask_loss
        losses['loss'] += mask_loss

    # def add_l2reg_losses(self, outputs, losses, model_name='instance_motion'):
    #     for param in self.models[model_name].parameters():
    #         if 'weight' in param:
    #             l2_reg += torch.norm(param)
    #     loss += lambda * l2_reg 

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            # if self.opt.geometric_loss:
            #     depth_input = [inputs["color_aug", 0, 0], inputs["color_aug", -1, 0], inputs["color_aug", 1, 0]]
            #     features = self.models["encoder"](torch.cat(depth_input, 0))
            # else:


            if self.opt.SIG:
                disp_net_input = torch.cat([
                    inputs["color_aug", 0, 0], 
                    inputs["sem_seg_one_hot", 0, 0],
                    inputs["ins_id_seg_to_edge", 0, 0]], 1)
                features = self.models["encoder"](disp_net_input)
            else:
                features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))
    
        # stage1: predict tgt_pred1
        self.generate_images_pred(inputs, outputs)

        if self.opt.instance_pose:
            # stage2: add an instance_pose_net to predict instance pose for each ins. 
            # Then predict tgt_pred2
            # self.synthesize_layer(inputs, outputs)
            self.generate_pred2_by_ins_pose(inputs, outputs)

        # if self.opt.instance_motion:
            # stage3: predict instance non-rigid-motion for each ins.
            # Then predict tgt_pred3, which is the final prediction.
            # self.generate_pred3_by_ins_non_rigid(inputs, outputs)

        losses = self.compute_losses(inputs, outputs)

        if self.opt.instance_pose:
            _, _, ins_losses = self.compute_instance_losses(inputs, outputs)
            # weight_fg, weight_bg, ins_losses = self.compute_instance_losses(inputs, outputs)
            # weight_fg, weight_bg, ins_losses = self.add_instance_losses(inputs, outputs)
            if ins_losses['ins_loss'].detach().cpu().numpy() == np.nan:
                print('nan')
                input()

            losses['ins_loss'] = ins_losses['ins_loss']
            bg_loss = losses['loss'].clone()
            fg_loss = losses['ins_loss']
            losses['bg_loss'] = bg_loss

            if self.opt.weight_fg is not None:
                losses['loss'] = (1-self.opt.weight_fg) * bg_loss + self.opt.weight_fg * fg_loss

            if self.opt.mask_loss_weight is not None:
                self.add_mask_losses(outputs, losses)

            # if self.opt.instance_motion:
            #     if self.opt.l2_regularization:
            #         self.add_l2reg_losses(outputs, losses)

        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                if True:
                    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if True:
                        if self.opt.disable_pose_invert:
                            pose_inputs = [pose_feats[0], pose_feats[f_i]]
                        else:
                            if f_i < 0:
                                pose_inputs = [pose_feats[f_i], pose_feats[0]]
                            else:
                                pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation

                    if self.opt.disable_pose_invert:
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=False)
                    else:
                        # Invert the matrix if the frame id is negative
                        outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                            axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        bs = self.opt.batch_size
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            # if self.opt.geometric_loss:
            if True:
                outputs[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                # fwd warping: T (tgt->src), warp t+1/t-1 to t
                cam_points = self.backproject_depth[source_scale](
                    outputs[("depth", 0, scale)], inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        if True:
            abs_diff = torch.abs(target - pred)
            l1_loss = abs_diff.mean(1, True)

            if self.opt.no_ssim:
                reprojection_loss = l1_loss
            else:
                ssim_loss = self.ssim(pred, target).mean(1, True)
                reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        if self.opt.SIG_ignore_fg_loss:
            tgt_dynamic_layer = inputs[("ins_id_seg_bk", 0, 0)].float()
            tgt_backgroud_layer = 1 - tgt_dynamic_layer  # [bs, 192, 640]

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += torch.randn(
                    identity_reprojection_loss.shape).cuda() * 0.00001

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                # [bs, 4, 192, 640] -> [bs, 192, 640]
                to_optimise, idxs = torch.min(combined, dim=1)

            if self.opt.SIG_ignore_fg_loss:
                to_optimise *= tgt_backgroud_layer

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            
            # --------------------------------------------------------------------
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            if self.opt.second_order_disp:
                smooth_loss = get_sec_smooth_loss(norm_disp, color)
            else:
                smooth_loss = get_smooth_loss(norm_disp, color)
            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            
            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def get_ins_bbox(self, inputs, frame_id, scale):
        # TODO: [bs, k ,4] -> [[K, 4]*bs], list of [k, 4], length = bs
        ins_RoI_bbox_frame_id = inputs[("ins_RoI_bbox", frame_id, scale)] #[bs, k=4 ,4]
        ins_RoI_bbox_list_frame_id = [x.squeeze(0) for x in list(ins_RoI_bbox_frame_id.split(1, dim=0))]
        return ins_RoI_bbox_list_frame_id

    def compute_IOU(self, mask1, mask2):
        """
        mask1: b, 1, h, w
        """
        inter = mask1 * mask2 # b,)
        outer = 1 - (1-mask1) * (1-mask2) # b,
        IOU = inter.sum([2, 3]).float() / (outer.sum([2, 3]).float()+1e-3) # b,
        return IOU

    def compute_outer(self, mask1, mask2):
        """
        mask1: b, 1, h, w
        """
        outer = 1 - (1-mask1) * (1-mask2) # b,
        return outer

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


    def transform_mul(self, A, B):
        C = A.clone()
        C[:, :3,3:] = torch.bmm(A[:, :3,:3], B[:, :3,3:]) + A[:, :3,3:]
        C[:, :3,:3] = torch.bmm(A[:, :3,:3], B[:, :3,:3])
        return C

    def generate_pred2_by_ins_pose(self, inputs, outputs):
        # some definitions
        scale = 0
        inv_K = inputs[("inv_K", scale)]
        K = inputs[("K", scale)]
        img0_aug = inputs["color_aug", 0, scale]
        img0 = inputs["color", 0, scale]
        img0_feature = self.models["instance_pose_encoder"](img0_aug)[-1] # [bs, 512, 6, 20]

        # compute tgt depth
        disp = outputs[("disp", scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        _, depth0 = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth) # min_depth = 0.1, max_depth = 100

        for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
            # ------------------------stage2------------------------
            # predict an insRT for each instance
            # ------------------------------------------------------
            # some definitions
            T_static = outputs[("cam_T_cam", 0, frame_id)] # [bs, 4, 4]
            cam_points = self.backproject_depth[scale](
                depth0, inv_K) # cam_points of frame 0, [12, 4, 122880]
            pix_coords = self.project_3d[scale](
                cam_points, K, T_static)
            
            # cam_coords0_pred = torch.matmul(T_static[:, :3, :], cam_points) # bs, 3, 122880
            # ones = nn.Parameter(torch.ones(self.opt.batch_size, 1, self.opt.height * self.opt.width),
            #                      requires_grad=False).to(self.device)
            # cam_coords0_pred = torch.cat([cam_coords0_pred, ones], 1) # bs, 4, 122880
            
            img1 = inputs["color", frame_id, scale]
            img0_pred = outputs[("color", frame_id, scale)]
            mask1 = torch.sum(inputs[("ins_id_seg", frame_id, scale)][:, 1:, :, :], 1).unsqueeze(1).float()
            mask0_pred = F.grid_sample(mask1, pix_coords, padding_mode="border")
            mask0_pred = self.filter_mask(mask0_pred)

            # compute image feature and crop ROI feature
            img0_pred_feature = self.models["instance_pose_encoder"](img0_pred)[-1] # [bs, 512, 6, 20]

            # FIXME: define the final image and mask
            # img0_pred2_base = img0_pred.clone()
            img0_pred2_base = torch.zeros_like(img0_pred, requires_grad=True)   # final image
            mask0_pred2_base = torch.zeros([self.opt.batch_size, 1, self.opt.height, self.opt.width], requires_grad=True).to(self.device) # bs, 1, 192, 640
            
            # init final depth map
            if self.opt.use_depth_ordering:
                depth0_pred2_base = 80.0 * torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width], requires_grad=True).to(self.device) # bs, 1, 192, 640

            instance_K_num = inputs[("ins_id_seg", frame_id, scale)].shape[1] - 1 
            T_dynamic_list = list()
            for ins_id in range(instance_K_num-1, -1, -1): # benefit to large instance
                # use T_static to transform mask of each ins
                img1_ins_mask = inputs[("ins_id_seg", frame_id, scale)][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
                img0_pred_ins_mask = F.grid_sample(img1_ins_mask, pix_coords, padding_mode="border") #[b, 1, h, w]
                img0_pred_ins_mask = self.filter_mask(img0_pred_ins_mask)

                # crop ins feature of img0 and img0_pred
                img0_pred_ins_bbox = self.extract_bbox_from_mask(img0_pred_ins_mask)
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
                T_dynamic_list.append(T_dynamic)
                
                # step7: predict ins
                if self.opt.use_depth_ordering:
                    # ins_pix_coords, img_z = self.project_3d[scale](cam_points, K, T_dynamic, return_z=True)
                    ins_pix_coords, img_z = self.project_3d[scale](cam_points, K, torch.bmm(T_static, T_dynamic), return_z=True)
                    # ins_pix_coords, img_z = self.project_3d[scale](cam_points, K, self.transform_mul(T_dynamic, T_static), return_z=True)
                else:
                    # ins_pix_coords = self.project_3d[scale](cam_points, K, T_dynamic)
                    ins_pix_coords = self.project_3d[scale](cam_points, K, torch.bmm(T_static, T_dynamic))
                    # ins_pix_coords = self.project_3d[scale](cam_points, K, self.transform_mul(T_dynamic, T_static))

                #step8: predict frame 0 from frame 1 based on T_dynamic and T_static
                # img0_pred2_ins = F.grid_sample(img0_pred, ins_pix_coords, padding_mode="border")
                # img0_pred2_ins_mask = F.grid_sample(img0_pred_ins_mask, ins_pix_coords, padding_mode="border") # [bs, 1, 192, 640]
                img0_pred2_ins = F.grid_sample(img1, ins_pix_coords, padding_mode="border")
                img0_pred2_ins_mask = F.grid_sample(img1_ins_mask, ins_pix_coords, padding_mode="border") # [bs, 1, 192, 640]
                img0_pred2_ins_mask = self.filter_mask(img0_pred2_ins_mask)

                # ------------------------stage3------------------------
                # version 1
                # predict instance motion for each instance
                # ------------------------------------------------------
                if self.opt.instance_motion and self.opt.instance_motion_v1:
                    # img0_ins_feature
                    img0_pred2_ins_bbox = self.extract_bbox_from_mask(img0_pred2_ins_mask)
                    img0_pred2_feature = self.models["instance_pose_encoder"](img0_pred2_ins)[-1] # [bs, 512, 6, 20]
                    img0_pred2_ins_feature = torchvision.ops.roi_align(img0_pred2_feature, img0_pred2_ins_bbox, output_size=(6,20))

                    # input ins_pose_net and predict ins_motion
                    if self.opt.disable_inspose_invert:
                        ins_motion_inputs = [img0_ins_feature, img0_pred2_ins_feature]
                    else:
                        if frame_id < 0:
                            ins_motion_inputs = [img0_pred2_ins_feature, img0_ins_feature]
                        else:
                            ins_motion_inputs = [img0_ins_feature, img0_pred2_ins_feature]
                    
                    # compute non rigid motion
                    ins_motion_inputs = torch.cat(ins_motion_inputs, 1)
                    non_rigid_motion_map = self.models["instance_motion"](ins_motion_inputs)

                    # add non rigid motion
                    if self.opt.max_speed is not None:
                        cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map, 
                                                                            self.opt.max_speed, self.opt.max_speed, self.opt.max_speed)
                    else:
                        cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map)

                    pix_coords_stage3 = self.project_3d[scale](
                        cam_points0_pred_stage3, K, torch.eye(4).to(self.device))
                    
                    # step5: warp new image
                    img0_pred2_ins = F.grid_sample(
                                            img0_pred2_ins,
                                            pix_coords_stage3,
                                            padding_mode="border")
                    img0_pred2_ins_mask = F.grid_sample(
                                            img0_pred2_ins_mask,
                                            pix_coords_stage3,
                                            padding_mode="border")
                    img0_pred2_ins_mask = self.filter_mask(img0_pred2_ins_mask)
                
                #step8.5: use IOU value to filter invalid points
                if self.opt.iou_thres is not None:
                    img0_ins_mask = inputs[("ins_id_seg", 0, scale)][:, ins_id+1, :, :].unsqueeze(1).float()
                    ins_IOU = self.compute_IOU(img0_ins_mask, img1_ins_mask) # [b, 1]
                    IOU_mask = ins_IOU > self.opt.iou_thres # [b, 1]
                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask * IOU_mask.float() # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

                #step8.6: use diff between t_pred and t_gt to eliminate relative static area
                if self.opt.roi_diff_thres is not None:
                    roi_abs = torch.abs(outputs[("color", frame_id, scale)] * img0_pred_ins_mask - inputs["color", 0, scale] * img0_pred_ins_mask)
                    # roi_abs: bs, 3, 192, 640
                    roi_sum = torch.sum(roi_abs, dim=[1, 2, 3]) # bs,
                    mask_sum = torch.sum(img0_pred_ins_mask, dim=[1, 2, 3]) # bs,
                    roi_diff = roi_sum.float() * 1.0 / (mask_sum.float()+1e-3) # bs,
                    roi_diff = roi_diff.unsqueeze(1) # [bs, 1]
                    roi_mask = roi_diff > self.opt.roi_diff_thres # [bs, 1]

                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask * roi_mask.float() # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

                #step9: predict image
                # img0_pred_final:[bs, 3, 192, 640], img0_pred2_ins_mask: [bs, 1, 192, 640], ins_pix_coords: [1, 192, 640, 2]
                if self.opt.use_depth_ordering:
                    # img_z: bs, 1, 192, 640
                    ins_z = img_z * img0_pred2_ins_mask
                    ins_z_mean = torch.sum(ins_z, [1, 2, 3]).float() / (torch.sum(img0_pred2_ins_mask, [1, 2, 3]).float()+1e-3)
                    depth0_pred_mean = torch.sum(depth0_pred2_base*img0_pred2_ins_mask, [1, 2, 3]).float() / (torch.sum(img0_pred2_ins_mask, [1, 2, 3]).float()+1e-3)
                    insz_less_than_depth = (ins_z_mean<depth0_pred_mean).unsqueeze(1) # bs, 1

                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask * insz_less_than_depth.float() # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

                    depth0_pred2_base = torch.add(depth0_pred2_base*(1-img0_pred2_ins_mask), img_z*img0_pred2_ins_mask)

                if self.opt.eval_flow_filter_size:
                    mask_sum = torch.sum(img0_pred_ins_mask, [2, 3])
                    mask_valid = (mask_sum >= 4900)
                    
                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, -1) # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask * mask_valid.float() # [b, 1x192x640]
                    img0_pred2_ins_mask = img0_pred2_ins_mask.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)

                
                mask0_pred2_base = torch.add(mask0_pred2_base*(1-img0_pred2_ins_mask), img0_pred2_ins_mask)
                if self.opt.eval_flow_mask_outer:
                    cur_img_mask = self.compute_outer(img0_pred2_ins_mask, img0_pred_ins_mask).clone()
                    img0_pred2_base = torch.add(img0_pred2_base*(1-cur_img_mask), img0_pred2_ins*cur_img_mask)
                else:    
                    img0_pred2_base = torch.add(img0_pred2_base*(1-img0_pred2_ins_mask), img0_pred2_ins*img0_pred2_ins_mask)
                # save for vis
                outputs[("T_dynamic", frame_id, ins_id)] = T_dynamic
            
            # FIXME: img0_pred2 = mask0_pred2_base * img0_pred2_base + (1-mask0_pred2_base) * img1            
            # FIXME:
            mask0_pred2 = mask0_pred2_base.clone() # + (1-mask0_pred2_base) * mask0_pred
            if self.opt.eval_flow_mask_outer:
                cur_img_mask = self.compute_outer(mask0_pred2_base, mask0_pred).clone()
                img0_pred2 = cur_img_mask * img0_pred2_base + (1-cur_img_mask) * img0_pred
            else:    
                img0_pred2 = mask0_pred2_base * img0_pred2_base + (1-mask0_pred2_base) * img0_pred
            img0_pred_latest = img0_pred2.clone()
            mask0_pred_latest = mask0_pred2.clone()

            # ------------------------stage3------------------------
            # version 2
            # predict a motion map for the whole image/instance area
            # ------------------------------------------------------
            if self.opt.instance_motion and self.opt.instance_motion_v2:
                if self.opt.warping_error_thres is not None:
                    # if warping error of img0_pred2 is larger, than use img0_pred instead
                    error_pred = torch.mean(torch.abs(img0_pred2 - img0), [1, 2, 3])
                    warping_error_thres = self.opt.warping_error_thres * torch.ones_like(error_pred)
                    mask_valid = (error_pred > warping_error_thres).unsqueeze(1).float()

                    mask_warping_error = torch.ones_like(mask0_pred2)
                    mask_warping_error = mask_warping_error.view(self.opt.batch_size, -1) # [b, 1x192x640]
                    mask_warping_error = mask_warping_error * mask_valid.float() # [b, 1x192x640]
                    mask_warping_error = mask_warping_error.view(self.opt.batch_size, 1, self.opt.height, self.opt.width)
                                 
                if self.opt.train_filter_warping_error:
                    # if warping error of img0_pred2 is larger, than use img0_pred instead
                    error_pred = torch.sum(torch.abs(img0_pred - img0), 1)
                    error_pred2 = torch.sum(torch.abs(img0_pred2_base - img0), 1)
                    mask_valid = (error_pred > error_pred2).unsqueeze(1).float()

                    img0_pred2 = mask_valid * img0_pred2_base + (1-mask_valid) * img0_pred
                    mask0_pred2 = mask_valid * mask0_pred2_base + (1-mask_valid) * mask0_pred
                
                if self.opt.use_fg_stage3:
                    mask0 = torch.sum(inputs[("ins_id_seg", 0, scale)][:, 1:, :, :], 1).unsqueeze(1).float()
                    img0_feature = self.models["instance_pose_encoder"](img0 * mask0)[-1] # [bs, 512, 6, 20]
                    img0_pred2_feature = self.models["instance_pose_encoder"](img0_pred2 * mask0_pred2)[-1] # [bs, 512, 6, 20]
                else:
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
                
                # add a non rigid motion
                if self.opt.max_speed is not None:
                    cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map, 
                                                                        self.opt.max_speed, self.opt.max_speed, self.opt.max_speed)
                else:
                    cam_points0_pred_stage3 = self.add_non_rigid_motion(cam_points, non_rigid_motion_map,
                                                                        self.opt.min_depth, self.opt.max_depth)
                
                img0_pred3_base = torch.zeros_like(img0_pred, requires_grad=True)   # final image
                mask0_pred3_base = torch.zeros([self.opt.batch_size, 1, self.opt.height, self.opt.width], requires_grad=True).to(self.device) # bs, 1, 192, 640
                
                # pix_coords_stage3 = self.project_3d[scale](
                #     cam_points0_pred_stage3, K, torch.eye(4).to(self.device))
                # # step5: warp to a new image
                # img0_pred3_base = F.grid_sample(
                #                         img0_pred2,
                #                         pix_coords_stage3,
                #                         padding_mode="border")
                # mask0_pred3_base = F.grid_sample(
                #                         mask0_pred2,
                #                         pix_coords_stage3,
                #                         padding_mode="border")
                # mask0_pred3_base = self.filter_mask(mask0_pred3_base)
                for ins_id, T_dynamic in enumerate(T_dynamic_list):
                    img1_ins_mask = inputs[("ins_id_seg", frame_id, scale)][:, ins_id+1, :, :].unsqueeze(1).float() # [b, 1, h, w]
                    pix_coords_stage3 = self.project_3d[scale](
                        cam_points0_pred_stage3, K, torch.bmm(T_static, T_dynamic) )
                    # step5: warp to a new image
                    img0_pred3_ins = F.grid_sample(
                                            img1,
                                            pix_coords_stage3,
                                            padding_mode="border")
                    img0_pred3_ins_mask = F.grid_sample(
                                            img1_ins_mask, 
                                            pix_coords_stage3, 
                                            padding_mode="border") # [bs, 1, 192, 640]
                    img0_pred3_ins_mask = self.filter_mask(img0_pred3_ins_mask)

                    img0_pred3_base = torch.add(img0_pred3_base*(1-img0_pred3_ins_mask), img0_pred3_ins*img0_pred3_ins_mask)
                    mask0_pred3_base = torch.add(mask0_pred3_base*(1-img0_pred3_ins_mask), img0_pred3_ins_mask)
                    
                # generate a new image
                if self.opt.predict_img_motion:
                    img0_pred3 = img0_pred3_base
                    mask0_pred3 = mask0_pred3_base
                else:
                    mask0_pred3 = mask0_pred3_base #+ (1-mask0_pred3_base) * mask0_pred2
                    if self.opt.eval_flow_mask_outer:
                        cur_img_mask = self.compute_outer(mask0_pred3, mask0_pred2).clone()
                        img0_pred3 = cur_img_mask * img0_pred3_base + (1-cur_img_mask) * img0_pred2
                    else:
                        # img0_pred_latest = mask0_pred_final * img0_pred_final + (1-mask0_pred_final) * img0_pred
                        img0_pred3 = mask0_pred3_base * img0_pred3_base + (1-mask0_pred3_base) * img0_pred2
                    
                if self.opt.warping_error_thres:
                    img0_pred_latest = img0_pred3 * mask_warping_error + img0_pred2 * (1-mask_warping_error)
                    mask0_pred_latest = mask0_pred3 * mask_warping_error + mask0_pred2 * (1-mask_warping_error)
                else:
                    img0_pred_latest = img0_pred3.clone()
                    mask0_pred_latest = mask0_pred3.clone()

                # save for vis
                outputs[("img0_pred2", frame_id, scale)] = img0_pred2
                outputs[("img0_pred3", frame_id, scale)] = img0_pred3
                outputs[("delta_x_mean", frame_id, scale)] = torch.mean(non_rigid_motion_map[0])

            # save for vis
            outputs[("img0_pred", frame_id, scale)] = img0_pred
            outputs[("color", frame_id, scale)] = img0_pred_latest
            # outputs[("foreground", frame_id, scale)] = img0_pred_latest * mask0_pred_latest
            outputs[("warped_mask", frame_id, scale)] = mask0_pred_latest
            
        # save for vis
        mask0 = torch.sum(inputs[("ins_id_seg", 0, scale)][:, 1:, :, :], 1).unsqueeze(1).float()
        outputs[("cur_mask", 0, scale)] = mask0
        outputs[("foreground", 0, scale)] = inputs["color", 0, scale] * mask0
    

    def extract_bbox_from_mask(self, ins_warp_mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        # ins_warp_mask: [bs, 1, 192, 640]
        mask = ins_warp_mask.squeeze(1)
        ins_warp_bbox = []
        for bs_idx in range(mask.shape[0]):
            #idx_mask = mask[bs_idx, :, :].uint8()#.detach().cpu().numpy()
            # Bounding box.
            idx_mask = mask[bs_idx, :, :].type(torch.uint8)
            horizontal_indicies = torch.where(torch.any(idx_mask, axis=0))[0]
            vertical_indicies = torch.where(torch.any(idx_mask, axis=1))[0]
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
                # No mask for this instance. Might happen due to
                # resizing or cropping. Set bbox to zeros
                x1, y1, x2, y2 = 0, 0, 639, 191

            ins_warp_bbox.append(torch.Tensor([[np.float(x1)/32.0, np.float(y1)/32.0, np.float(x2)/32.0, np.float(y2)/32.0]]).to(self.device))
            #ins_warp_bbox.append([[x1, y1, x2, y2]])
            #ins_warp_bbox.append(torch.Tensor([[x1, y1, x2, y2]]).to(self.device))

        # list of [1,4]
        return ins_warp_bbox

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
            
    def compute_instance_losses(self, inputs, outputs):
        """loss of dynamic region"""

        def get_ins_smooth_loss(disp_with_mask, img_with_mask, total_mask):
            """Computes the smoothness loss for an instance
            The color image is used for edge-aware smoothness
            """
            disp = disp_with_mask
            img = img_with_mask

            grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
            grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

            grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
            grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

            grad_disp_x *= torch.exp(-grad_img_x)
            grad_disp_y *= torch.exp(-grad_img_y)

            grad_disp_x_mean = grad_disp_x.sum([1, 2, 3]).float() / (total_mask.sum([1, 2, 3]).float() + 1e-3)
            grad_disp_y_mean = grad_disp_y.sum([1, 2, 3]).float() / (total_mask.sum([1, 2, 3]).float() + 1e-3)

            # smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()
            # smooth_loss = grad_disp_x_mean.mean() + grad_disp_y_mean.mean()
            smooth_loss = torch.sum(grad_disp_x_mean).float() / (torch.sum(total_mask.sum([1, 2, 3]) > 0).float() + 1e-3) \
                            + torch.sum(grad_disp_y_mean).float() / (torch.sum(total_mask.sum([1, 2, 3]) > 0).float() + 1e-3)
            
            return smooth_loss
        
        def get_ins_reproj_loss(outputs, scale):
            tgt_foreground = outputs[("foreground", 0, scale)]
            total_mask = outputs[("cur_mask", 0, scale)]

            reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
                # pred_foreground = outputs[("foreground", frame_id, scale)]
                pred_foreground = outputs[("color", frame_id, scale)] * total_mask
                reprojection_losses.append(self.compute_reprojection_loss(pred_foreground, tgt_foreground))

            combined = torch.cat(reprojection_losses, 1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            # reproj_loss = to_optimise.sum([1, 2]) / (total_mask.sum([1, 2, 3])+1e-3) # [b]
            reproj_loss = (to_optimise*total_mask.squeeze()).sum([1, 2]).float() / (total_mask.sum([1, 2, 3]).float() + 1e-3)
            reproj_loss = torch.sum(reproj_loss).float() / (torch.sum(total_mask.sum([1, 2, 3]) > 0).float() + 1e-3)

            return reproj_loss

        losses = {}
        scale = 0

        # compute instance smooth loss
        disp = outputs[("disp", scale)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        total_mask = outputs[("cur_mask", 0, scale)]
        smooth_loss = get_ins_smooth_loss(norm_disp*total_mask, inputs[("color", 0, scale)]*total_mask, total_mask)
        smooth_loss =  self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

        # compute instance reprojection loss
        reproj_loss = get_ins_reproj_loss(outputs, scale)

        # summarize)
        losses["ins_loss/{}_smooth_loss".format(scale)] = smooth_loss
        losses["ins_loss/{}_reproj".format(scale)] = reproj_loss
        losses["ins_loss_{}".format(scale)] = reproj_loss + smooth_loss
        losses["ins_loss"] = reproj_loss + smooth_loss

        # FIXME:
        weight_fg = total_mask.sum() / (total_mask.nelement() + 1e-3)
        weight_bg = 1 - weight_fg

        return weight_fg, weight_bg, losses
    '''
    def compute_instance_losses_ori(self, inputs, outputs):
        """loss of dynamic region"""

        losses = {}
        scale = 0

        total_mask = outputs[("cur_mask", 0, scale)]

        if total_mask.sum() < 1:
            # TODO: check
            losses["ins_loss"] = torch.zeros(1, requires_grad=True).mean().to(self.device)
            weight_fg = 0
            weight_bg = 1 - weight_fg
        else:
            color = inputs[("color", 0, scale)]
            disp = outputs[("disp", scale)]
            disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)

            if self.opt.second_order_disp:
                smooth_loss = get_sec_smooth_loss(norm_disp, color)
            else:
                smooth_loss = get_smooth_loss(norm_disp, color)
            smooth_loss =  self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            losses["ins_loss/{}_smooth_loss".format(scale)] = smooth_loss

            reprojection_losses = []
            tgt_dynamic = outputs[("foreground", 0, scale)]
            for frame_id in self.opt.frame_ids[1:]: # [-1, 1]
                pred_dynamic = outputs[("foreground", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred_dynamic, tgt_dynamic))

            combined = torch.cat(reprojection_losses, 1)
            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            reproj_loss = to_optimise.mean()

            losses["ins_loss/{}_reproj".format(scale)] = reproj_loss
            losses["ins_loss_{}".format(scale)] = reproj_loss + smooth_loss

            losses["ins_loss"] = reproj_loss + smooth_loss

            weight_fg = total_mask.sum() / total_mask.nelement()
            weight_bg = 1 - weight_fg

        return weight_fg, weight_bg, losses
    '''
    # def log_time(self, batch_idx, duration, loss):
    def log_time(self, batch_idx, duration, loss, ins_loss=None, bg_loss=None, mask_loss=None):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0

        if ins_loss is not None:
            if mask_loss is not None:
                print_string = "epoch {:>3} | batch {:>6}" + \
                    " | loss: {:.5f}| ins_loss: {:.5f} | bg_loss: {:.5f} | mask_loss: {:.5f} | time elapsed: {} | time left: {}"

                print(print_string.format(self.epoch, batch_idx, loss, ins_loss, bg_loss, mask_loss,
                                        sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
            else:
                print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                    " | loss: {:.5f}| ins_loss: {:.5f} | bg_loss: {:.5f} | time elapsed: {} | time left: {}"

                print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, ins_loss, bg_loss,
                                        sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))
        else:
            print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                " | loss: {:.5f} | time elapsed: {} | time left: {}"

            print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                    sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, add_image=False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        # for l, v in losses.items():
        #     writer.add_scalar("{}".format(l), v, self.step)

        if add_image == True:
            for j in range(min(1, self.opt.batch_size)):  # write a maxmimum of four images
                # for s in self.opt.scales:
                for s in [0]:
                    for frame_id in self.opt.frame_ids:
                        writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                        # if s == 0 and frame_id != 0:
                        #     writer.add_image(
                        #         "color_pred_{}_{}/{}".format(frame_id, s, j),
                        #         outputs[("color", frame_id, s)][j].data, self.step)
                        if frame_id != 0:
                            writer.add_image(
                                "img0_pred_{}_{}/{}".format(frame_id, s, j),
                                outputs[("img0_pred", frame_id, s)][j].data, self.step)

                        if self.opt.instance_pose and self.opt.instance_motion:# and self.opt.instance_motion_v2:
                            # if frame_id == 0:
                            #     writer.add_image(
                            #         "outputs_foreground_{}_{}/{}".format(frame_id, s, j),
                            #         outputs[("foreground", frame_id, 0)][j].data, self.step)

                            if frame_id != 0:
                                writer.add_image(
                                    "img0_pred2_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("img0_pred2", frame_id, s)][j].data, self.step)

                                writer.add_image(
                                    "img0_pred3_{}_{}/{}".format(frame_id, s, j),
                                    outputs[("img0_pred3", frame_id, s)][j].data, self.step)

                                # writer.add_image(
                                #     "color_pred_ori_{}_{}/{}".format(frame_id, s, j),
                                #     outputs[("color_ori", frame_id, s)][j].data, self.step)

                                # writer.add_image(
                                #     "outputs_foreground_{}_{}/{}".format(frame_id, s, j),
                                #     outputs[("foreground", frame_id, 0)][j].data, self.step)

                                # writer.add_image(
                                #     "color_diff_{}_{}/{}".format(frame_id, s, j),
                                #     outputs[("color_diff", frame_id, 0)][j].data, self.step)

                                # writer.add_image(
                                #     "warped_mask_{}_{}/{}".format(frame_id, s, j),
                                #     outputs[("warped_mask", frame_id, 0)][j].data, self.step)

                                # writer.add_image(
                                #     "mask_{}_{}/{}".format(frame_id, s, j),
                                #     outputs[("mask", frame_id, 0)][j].data, self.step)

                                '''
                                outputs[("color_ori", frame_id, scale)] = color_ori
                                outputs[("color_diff", frame_id, scale)] = color_new - color_ori
                                outputs[("color", frame_id, scale)] = color_new
                                outputs[("foreground", frame_id, scale)] = img0_pred_final
                                outputs[("warped_mask", frame_id, scale)] = mask0_pred_final
                                '''

                    writer.add_image(
                        "disp_{}/{}".format(s, j),
                        normalize_image(outputs[("disp", s)][j]), self.step)
                        
    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

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

        # if not self.opt.instance_pose and not self.opt.instance_motion:
        try:
            # loading adam state
            optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
            # if os.path.isfile(optimizer_load_path):
            if os.path.exists(optimizer_load_path) == True:
                print("Loading Adam weights")
                optimizer_dict = torch.load(optimizer_load_path)
                self.model_optimizer.load_state_dict(optimizer_dict)
        except Exception as e:
            print("Cannot find Adam weights so Adam is randomly initialized")
