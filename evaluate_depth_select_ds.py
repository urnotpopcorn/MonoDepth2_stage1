from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate(opt, weight_id=None):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    if opt.ext_disp_to_eval is None:
        if weight_id is not None:
            load_weights_folder = os.path.join(opt.log_dir, 
                opt.load_weights_folder, "models", "weights_" + str(weight_id))
        else:
            load_weights_folder = opt.load_weights_folder

        assert os.path.isdir(load_weights_folder), \
            "Cannot find a folder at {}".format(load_weights_folder)

        # print("-> Loading weights from {}".format(load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, "drivingstereo_eigen", "test_files.txt"))
        encoder_path = os.path.join(load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)

        img_ext = ".jpg"

        dataset = datasets.DSRAWDataset(opt.data_path, filenames,
                                        encoder_dict['height'], encoder_dict['width'],
                                        [0], len(opt.scales), is_train=False, img_ext=img_ext,
                                        opt=opt, mode="test")
        # bs = 16
        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        
        if opt.SIG:
            encoder = networks.ResnetEncoder(opt.num_layers, False, mode = "SIG", cur="depth")
        else:
            encoder = networks.ResnetEncoder(opt.num_layers, False)
        
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, opt.scales)

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []

        # print("-> Computing predictions with size {}x{}".format(
        #     encoder_dict['width'], encoder_dict['height']))
        
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.SIG:
                    input_sem_seg_one_hot_float =  data["sem_seg_one_hot", 0, 0].cuda()
                    input_ins_id_seg_to_edge = data["ins_id_seg_to_edge", 0, 0].cuda()

                    # print("input_color: ", input_color.shape)
                    # print("input_sem_seg_one_hot_float: ", input_sem_seg_one_hot_float.shape)
                    # print("input_ins_id_seg_to_edge: ", input_ins_id_seg_to_edge.shape)

                    disp_net_input = torch.cat([
                            input_color, 
                            input_sem_seg_one_hot_float,
                            input_ins_id_seg_to_edge], 1)

                    output = depth_decoder(encoder(disp_net_input))

                else:
                    output = depth_decoder(encoder(input_color))

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        # print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

    if opt.save_pred_disps:
        output_path = os.path.join(
            load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        # print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    # print("-> Evaluating")

    # print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    if opt.eval_split == "drivingstereo_eigen_full":
        gt_depth_dir = "gt_depth_full"
    elif opt.eval_split == "drivingstereo_eigen_half":
        gt_depth_dir = "gt_depth_half"

    for i in range(pred_disps.shape[0]):
        gt_depth_file = os.path.join(splits_dir, "drivingstereo_eigen", gt_depth_dir, filenames[i]+".npy")
        gt_depth = np.load(gt_depth_file)
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        # TODO: crop number for DrivingStereo dataset
        # crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
        #                  0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        # crop_mask = np.zeros(mask.shape)
        # crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        # mask = np.logical_and(mask, crop_mask)

        # 0.40810811 * 375=153.04, 0.99189189 * 375=371.96
        # 0.03594771 * 1242==44.65, 0.96405229 * 1242==1197.35

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor # 1
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        err = compute_errors(gt_depth, pred_depth)
        # print(err)
        errors.append(err)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        # print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    # print("  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    # print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print(("{: 8.3f}\t" * 7).format(*mean_errors.tolist()))

if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    # weights_range = opts.depth_test_weights

    # for weight_id in weights_range:
    #     evaluate(opts, weight_id)
    evaluate(opts)
    torch.cuda.empty_cache()