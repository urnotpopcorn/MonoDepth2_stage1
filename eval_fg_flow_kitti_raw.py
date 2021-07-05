from __future__ import division
import cv2
import os
import numpy as np
import argparse
import sys
from tqdm import tqdm
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path + '/flow_tool/')
import flowlib as fl

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, help="Path to kitti stereo dataset")
parser.add_argument("--pred_dir",    type=str, help="Path to the flow prediction")
parser.add_argument("--ins_dir",    type=str, help="Path to the instance segmentation prediction")
parser.add_argument("--flow_type",    type=str, help="which flow to test, overall, fg, or bg")

args = parser.parse_args()
valid_class = [1,2,3,4,6,7,8,16,17,18,19,20,21,22,23,24]

def main():
    img_num = 697
    noc_epe = np.zeros(img_num, dtype=np.float)
    noc_acc = np.zeros(img_num, dtype=np.float)
    occ_epe = np.zeros(img_num, dtype=np.float)
    occ_acc = np.zeros(img_num, dtype=np.float)
    cnt_num = 0

    if args.flow_type == "all":
        eval_log = os.path.join(args.pred_dir, 'flow_result.txt')
    elif args.flow_type == "fg":
        eval_log = os.path.join(args.pred_dir, 'fg_flow_result.txt')
    elif args.flow_type == "bg":
        eval_log = os.path.join(args.pred_dir, 'bg_flow_result.txt')

    invalid_next_idx = [246, 316]
    invalid_pre_idx = [18, 25, 68, 82, 142, 153, 175, 200, 225, 268, 298, 327, 397, 422, 517, 525, 556, 591, 617, 641, 671, 676]
    
    # with open(eval_log, 'w') as el:
    if True:
        # for idx in tqdm(range(img_num)):
        for idx in range(img_num):
            try:
                if idx in invalid_next_idx:
                    continue
                # read groundtruth flow
                gt_occ_fn = os.path.join(args.dataset_dir, 'flow_gt/%.6d_10.png' % idx)
                gt_occ_flow = fl.read_flow(gt_occ_fn)
                # read predicted flow (in png format)
                # pred_flow_fn = os.path.join(args.pred_dir, 'png/%.6d_1_flo.png' % idx)
                pred_flow_fn = os.path.join(args.pred_dir, 'png/%.6d.png' % idx)
                # print("pred_flow_fn: ", pred_flow_fn)

                pred_flow = fl.read_flow(pred_flow_fn)
                # resize pred_flow to the same size as gt_flow
                dst_h = gt_occ_flow.shape[0]
                dst_w = gt_occ_flow.shape[1]
                pred_flow = fl.resize_flow(pred_flow, dst_w, dst_h)
                if args.flow_type == "all":
                    pass
                elif args.flow_type == "fg":
                    # print(os.path.join(args.ins_dir, '%.6d_10.npy' % idx))
                    dynamic_fg_ins = np.load(os.path.join(args.ins_dir, '%.6d_10.npy' % idx))[:,:,0]
                    objs = [dynamic_fg_ins==i for i in valid_class]
                    fg_mask = np.expand_dims(np.sum(objs, axis=0),2) * 1.0

                    eval_log = os.path.join(args.pred_dir, 'fg_flow_result.txt')
                    gt_occ_flow *= fg_mask
                    pred_flow *= fg_mask

                elif args.flow_type == "bg":
                    dynamic_fg_ins = np.load(os.path.join(args.ins_dir, '%.6d_10.npy' % idx))[:,:,0]
                    objs = [dynamic_fg_ins==i for i in valid_class]
                    fg_mask = 1 - np.expand_dims(np.sum(objs, axis=0),2) * 1.0

                    eval_log = os.path.join(args.pred_dir, 'bg_flow_result.txt')
                    gt_occ_flow *= fg_mask
                    pred_flow *= fg_mask
                # evaluation
                # (single_noc_epe, single_noc_acc) = fl.evaluate_kitti_flow(gt_noc_flow, pred_flow, None)
                try:
                    (single_occ_epe, single_occ_acc) = fl.evaluate_kitti_flow(gt_occ_flow, pred_flow, None)
                except Exception as e:
                    # print(idx)
                    # print(e)
                    single_occ_epe = 0
                    single_occ_acc = 1

                noc_epe[idx] = single_occ_epe
                noc_acc[idx] = single_occ_acc
                occ_epe[idx] = single_occ_epe
                occ_acc[idx] = single_occ_acc
                cnt_num += 1

                output_line = 'Flow %.6d Noc EPE = %.4f' + ' Noc ACC = %.4f' + ' Occ EPE = %.4f' + ' Occ ACC = %.4f\n';
                # el.write(output_line % (idx, noc_epe[idx], noc_acc[idx], occ_epe[idx], occ_acc[idx]))
            except Exception as e:
                print(e)
                continue

    noc_mean_epe = np.mean(noc_epe)
    noc_mean_acc = np.mean(noc_acc)
    occ_mean_epe = np.mean(occ_epe)
    occ_mean_acc = np.mean(occ_acc)

    # noc_mean_epe = np.sum(noc_epe) / cnt_num
    # noc_mean_acc = np.sum(noc_acc) / cnt_num
    # occ_mean_epe = np.sum(occ_epe) / cnt_num
    # occ_mean_acc = np.sum(occ_acc) / cnt_num

    print('Mean Noc EPE = %.4f ' % noc_mean_epe)
    print('Mean Noc ACC = %.4f ' % noc_mean_acc)
    print('Mean Occ EPE = %.4f ' % occ_mean_epe)
    print('Mean Occ ACC = %.4f ' % occ_mean_acc)

main()