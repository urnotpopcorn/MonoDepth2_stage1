import argparse
import math
#from datetime import datetime
import numpy as np
#import socket
#import importlib
import os
import sys
import glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#import pickle
#import pdb
#import utils.tf_util
#from utils.pointnet_util import *


def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.sum(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.sum(acc2)
    
    EPE = np.sum(error)
#     EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
#     EPE = np.sum(EPE)
    return EPE, acc1, acc2, error, gtflow_len


if __name__ == "__main__":
    len_cloud = 100000
    parser = argparse.ArgumentParser()
    parser.add_argument('--kitti_dataset', default='dataset/kitti_rm_ground', help='Dataset directory')
    FLAGS = parser.parse_args()
    KITTI_DATASET = FLAGS.kitti_dataset
    all_kitti = glob.glob(os.path.join(KITTI_DATASET, 'train/*.npz'))
    epe_total = 0
    batch_count = 0
    sample_count = 0
    all_pred = []
    all_label = []
    all_points = []
    
    for ki in all_kitti:
        x = np.load(ki)
        batch_label = []
        batch_data = []
        batch_mask = []
        ref_pc = x['pos1'][:, :3]
        ref_center = np.mean(ref_pc, 0)
        for i in range(0, len_cloud, 2048):
            if i+2048 < len(x['pos1']) and i+2048 < len(x['pos2']):
                pc1 = x['pos1'][i:i+2048, :3]
                pc2 = x['pos2'][i:i+2048, :3]
                gt = x['gt'][i:i+2048, :3]
                pc1 = pc1 - ref_center
                pc2 = pc2 - ref_center
                batch_data.append(np.concatenate([np.concatenate([pc1,
                                                                  pc2], axis=0), 
                                                  np.zeros((4096, 3))], axis=1)) # 4096, 6
                batch_label.append(gt)
        
        batch_data = np.array(batch_data) # n, 4096, 6
        print(batch_data[0, 0, :])
        print(batch_data[0, 2048, :])
        input()
        continue
        batch_label = np.array(batch_label) # n, 2048, 3

        epe, acc1, acc2, error, gt_label = scene_flow_EPE_np(pred_val, batch_label,
                                        np.ones(pred_val.shape, dtype=np.int32)[:,:,0])

        epe_total += epe
        sample_count += batch_data.shape[0]*(batch_data.shape[1]/2)
        batch_count += batch_data.shape[0]
        
        all_pred.append(pred_val)
        all_points.append(batch_data)
        all_label.append(batch_label)
        
            
    all_pred = np.array(all_pred)
    all_points = np.array(all_points)
    all_label = np.array(all_label)
    
    print (all_pred.shape, all_points.shape, all_label.shape)
    print('Num batches {} Average EPE {}'.format(sample_count,epe_total/sample_count))
    print ('eval mean EPE 3D: %f' % (epe_total / sample_count))
    
