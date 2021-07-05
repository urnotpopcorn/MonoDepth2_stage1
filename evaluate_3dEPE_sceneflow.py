import argparse
import math
from datetime import datetime
import numpy as np
# import tensorflow as tf
# import socket
# import importlib
import os
import sys
import glob
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
# import pickle

class SceneflowDataset():
    def __init__(self, root='dataset/kitti_rm_ground', npoints=16384, train=True):
        self.npoints = npoints
        self.root = root
        self.train = train
        self.datapath = glob.glob(os.path.join(self.root, '*.npz'))
        self.cache = {}
        self.cache_size = 30000

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, flow = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['pos1']
                pos2 = data['pos2']
                flow = data['gt']

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pos1, pos2, flow)

            n1 = pos1.shape[0]
            n2 = pos2.shape[0]
            if n1 >= self.npoints:
                sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, self.npoints - n1, replace=True)), axis=-1)
            if n2 >= self.npoints:
                sample_idx2 = np.random.choice(n2, self.npoints, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, self.npoints - n2, replace=True)), axis=-1)

            pos1_ = np.copy(pos1)[sample_idx1, :]
            pos2_ = np.copy(pos2)[sample_idx2, :]
            flow_ = np.copy(flow)[sample_idx1, :]

        color1 = np.zeros([self.npoints, 3])
        color2 = np.zeros([self.npoints, 3])
        mask = np.ones([self.npoints])

        return pos1_, pos2_, color1, color2, flow_, mask

    def __len__(self):
        return len(self.datapath)

def log_string(out_str):
    print(out_str)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT*2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 3))
    batch_mask = np.zeros((bsize, NUM_POINT))
    # shuffle idx to change point order (change FPS behavior)
    shuffle_idx = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)
    for i in range(bsize):
        pc1, pc2, color1, color2, vel, mask1 = dataset[idxs[i+start_idx]]

        batch_data[i,:NUM_POINT,:3] = pc1[shuffle_idx,:]
        batch_data[i,:NUM_POINT,3:] = color1[shuffle_idx,:]
        batch_data[i,NUM_POINT:,:3] = pc2[shuffle_idx,:]
        batch_data[i,NUM_POINT:,3:] = color2[shuffle_idx,:]
        batch_label[i,:,:] = vel[shuffle_idx,:]
        batch_mask[i,:] = mask1[shuffle_idx]
    return batch_data, batch_label, batch_mask

def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2

def eval_one_epoch(BATCH_SIZE, NUM_POINT, DATA, TEST_DATASET):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1) // BATCH_SIZE

    epe_3d_sum = 0
    acc_3d_sum = 0
    acc_3d_2_sum = 0

    log_string(str(datetime.now()))
    log_string('---- EVALUATION ----')

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    batch_mask = np.zeros((BATCH_SIZE, NUM_POINT))
    
    for batch_idx in range(num_batches):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label, cur_batch_mask = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            batch_mask = cur_batch_mask
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            batch_mask[0:cur_batch_size] = cur_batch_mask
        
        continue
        # FIXME:    
        pred_scene_flow = compute_scene_flow()
        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(pred_scene_flow, batch_label, batch_mask)
        print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f' % (epe_3d, acc_3d, acc_3d_2))

        if cur_batch_size==BATCH_SIZE:
            epe_3d_sum += epe_3d
            acc_3d_sum += acc_3d
            acc_3d_2_sum += acc_3d_2

    log_string('eval mean EPE 3D: %f' % (epe_3d_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval mean ACC 3D: %f' % (acc_3d_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval mean ACC 3D 2: %f' % (acc_3d_2_sum / float(len(TEST_DATASET)/BATCH_SIZE)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
    parser.add_argument('--data', default='dataset/kitti_rm_ground', help='Dataset directory')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

    BATCH_SIZE = FLAGS.batch_size
    NUM_POINT = FLAGS.num_point
    DATA = FLAGS.data
    TEST_DATASET = SceneflowDataset(DATA, npoints=NUM_POINT, train=False)
    # log_string('pid: %s'%(str(os.getpid())))
    # print(len(TEST_DATASET))
    eval_one_epoch(BATCH_SIZE, NUM_POINT, DATA, TEST_DATASET)
