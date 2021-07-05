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
    img_num = 150
    all_epe = np.zeros(img_num, dtype=np.float)
    all_acc = np.zeros(img_num, dtype=np.float)
    if args.flow_type == "all":
        eval_log = os.path.join(args.pred_dir, 'flow_result.txt')
    elif args.flow_type == "fg":
        eval_log = os.path.join(args.pred_dir, 'fg_flow_result.txt')
    elif args.flow_type == "bg":
        eval_log = os.path.join(args.pred_dir, 'bg_flow_result.txt')

    with open(eval_log, 'w') as el:
        # for idx in tqdm(range(img_num)):
        for idx in range(img_num):
            # read groundtruth flow
            gt_flow_fn = os.path.join(
                "dataset/RAFT/submit_ds/flow_select_150", 
                '%.6d_10.png' % idx)

            # print("gt_flow_fn: ", gt_flow_fn)
            gt_flow = fl.read_flow(gt_flow_fn)

            # read predicted flow (in png format)
            # pred_flow_fn = os.path.join(args.pred_dir, 'png/%.6d_1_flo.png' % idx)
            pred_flow_fn = os.path.join(args.pred_dir, 'png/%.6d.png' % idx)
            pred_flow = fl.read_flow(pred_flow_fn)

            # resize pred_flow to the same size as gt_flow
            dst_h = gt_flow.shape[0]
            dst_w = gt_flow.shape[1]
            pred_flow = fl.resize_flow(pred_flow, dst_w, dst_h)

            if args.flow_type == "all":
                pass
            elif args.flow_type == "fg":
                dynamic_fg_ins = np.load(os.path.join(args.ins_dir, '%.6d_10.npy' % idx))[:,:,0]

                objs = [dynamic_fg_ins==i for i in valid_class]
                fg_mask = np.expand_dims(np.sum(objs, axis=0),2) * 1.0

                eval_log = os.path.join(args.pred_dir, 'fg_flow_result.txt')
                gt_flow *= fg_mask
                pred_flow *= fg_mask

            elif args.flow_type == "bg":
                dynamic_fg_ins = np.load(os.path.join(args.ins_dir, '%.6d_10.npy' % idx))[:,:,0]
                objs = [dynamic_fg_ins==i for i in valid_class]
                fg_mask = 1 - np.expand_dims(np.sum(objs, axis=0),2) * 1.0

                eval_log = os.path.join(args.pred_dir, 'bg_flow_result.txt')
                gt_flow *= fg_mask
                pred_flow *= fg_mask

            # evaluation
            (single_noc_epe, single_noc_acc) = fl.evaluate_kitti_flow(gt_flow, pred_flow, None)
            (single_occ_epe, single_occ_acc) = fl.evaluate_kitti_flow(gt_flow, pred_flow, None)
            # if single_occ_epe > 100:
            #     print(idx)
            all_epe[idx] = single_noc_epe
            all_acc[idx] = single_noc_acc
            output_line = 'Flow %.6d All EPE = %.4f' + ' All ACC = %.4f'
            el.write(output_line % (idx, all_epe[idx], all_acc[idx]))

    all_mean_epe = np.mean(all_epe)
    all_mean_acc = np.mean(all_acc)

    print('Mean All EPE = %.4f ' % all_mean_epe)
    print('Mean All ACC = %.4f ' % all_mean_acc)

main()