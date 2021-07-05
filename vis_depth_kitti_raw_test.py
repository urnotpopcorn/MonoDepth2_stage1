from PIL import Image
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def view_idx(rgb_img, pred, vis_save_path):
    plt.figure(figsize=(12,12))
    plt.subplot(4,1,1)
    plt.imshow(rgb_img)
    plt.title("tgt")

    plt.subplot(4,1,2)
    plt.imshow(1.0/pred[0], cmap = "plasma")
    plt.title("Monodepth2")

    plt.subplot(4,1,3)
    plt.imshow(1.0/pred[1], cmap = "plasma")
    plt.title("Ours")

    plt.subplot(4,1,4)
    plt.imshow(pred[1]-pred[0], cmap = "bwr")
    plt.title("Diff")

    # plt.subplot(4,2,5)
    # plt.imshow(pred[2])
    # plt.title("M_416x128_no_SIG_3_disp_original")

    # plt.subplot(4,2,6)
    # plt.imshow(pred[3])
    # plt.title("M_416x128_SIG_104_disp_original")

    # plt.subplot(4,2,7)
    # plt.imshow(ins_img)
    # plt.title("SIG_ins_img")

    # plt.subplot(4,2,8)
    # plt.imshow(sem_img)
    # plt.title("SIG_sem_img")

    plt.show()
    plt.savefig(vis_save_path)
    plt.close()

def view_per_img(rgb_img, pred, vis_save_path):
    plt.figure()
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_rgb.jpg'), bbox_inches='tight', pad_inches=0.0)
    # plt.savefig('hah.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.imshow(1.0/pred[0], cmap = "plasma")
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_mono.jpg'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.imshow(1.0/pred[1], cmap = "plasma")
    plt.axis('off')
    plt.savefig(vis_save_path.replace('.jpg', '_ours.jpg'), bbox_inches='tight', pad_inches=0.0)
    plt.close()

def get_image_path(folder, frame_index, side):
    f_str = "{:010d}{}".format(frame_index, ".png") # '0000000268.jpg'

    # '/userhome/34/h3567721/projects/Depth/monodepth2/kitti_data/2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000268.jpg'
    # image_path = os.path.join(ori_kitti_data, folder, "image_0{}/data".format(side_map[side]), f_str)
    image_path = os.path.join(folder, "image_0{}/data".format(side_map[side]), f_str)

    return image_path

def get_ins_sem_path(folder, frame_index, side):
    f_str = "{:010d}{}".format(frame_index, ".npy") # '0000000268.jpg'

    # '/userhome/34/h3567721/projects/Depth/monodepth2/kitti_data/2011_09_26/2011_09_26_drive_0028_sync/image_02/data/0000000268.jpg'
    # image_path = os.path.join(ori_kitti_data, folder, "image_0{}/data".format(side_map[side]), f_str)
    image_path = os.path.join(folder, "image_0{}/data".format(side_map[side]), f_str)

    return image_path

def get_color(image_path):
    return Image.open(image_path).convert('RGB')

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def recale_depth(pred_disp, gt_depth):
    gt_height, gt_width = gt_depth.shape[:2]

    pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
    pred_depth = pred_disp
    pred_depth = 1 / pred_disp

    mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    
    pred_depth_mask = pred_depth[mask]
    gt_depth_mask = gt_depth[mask]
    
    # ratio = np.median(pred_depth_mask) / np.median(gt_depth_mask)
    ratio = np.median(gt_depth_mask) / np.median(pred_depth_mask)
    pred_depth *= ratio
    pred_depth_mask *= ratio

    # print(np.mean(pred_depth_mask))

    '''
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]
    
    ratio = np.median(gt_depth) / np.median(pred_depth)
    pred_depth *= ratio
    '''

    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    return pred_depth, gt_depth_mask, pred_depth_mask


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

if __name__ == "__main__":
    src_f_test_path = "splits/eigen/test_files.txt"

    with open(src_f_test_path, "r") as fd:
        src_f_test = fd.read().splitlines()

    side_map = {
        'l': "2",
        "r": "3"
    }

    rgb_path = "dataset/raw_data"

    gt_path = "splits/eigen/gt_depths.npz"
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    mono_path = "../MonoDepth2/models/mono_640x192/disps_eigen_split.npy"
    # pred_path = "log/stage123v2_monoori_woSIG_srcsample2_roi_y0_depth/models/weights_0/disps_eigen_split.npy"
    # save_dir = "output/vis_depth/Ours_Better"
    pred_path = sys.argv[1]
    save_dir = sys.argv[2]
    make_dir(save_dir)
    mono_disp = np.load(mono_path)
    pred_disp = np.load(pred_path)

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    seletex_idx = [320]
    
    for i, sig_file in tqdm(enumerate(src_f_test)):
        # 2011_10_03/2011_10_03_drive_0034_sync 3247 l
        # if i not in seletex_idx:
        #     continue
        # 2011_09_26_drive_0023_sync_252_l
        cond = '2011_09_26_drive_0023_sync' in sig_file and '252' in sig_file and 'l' in sig_file
        # if cond == False:
        #     continue
        # print(sig_file)
        
        line = sig_file.split(" ")
        folder = line[0]
        frame_index = int(line[1])
        side = line[2]

        image_path = get_image_path(folder, frame_index, side)
        rgb_img = get_color(os.path.join(rgb_path, image_path))
        # gt_depth = gt_depths[i]
        mono_depth, gt_depth_mask, mono_depth_mask = recale_depth(mono_disp[i], gt_depths[i])
        error_mono = compute_errors(gt_depth_mask, mono_depth_mask)

        pred_depth, gt_depth_mask, pred_depth_mask = recale_depth(pred_disp[i], gt_depths[i])
        error_pred = compute_errors(gt_depth_mask, pred_depth_mask)

        pred = [mono_depth, pred_depth]
        save_name = folder.split("/")[1] + "_" + str(frame_index) + "_" + side
        vis_save_path = os.path.join(save_dir, save_name + ".jpg")
        
        # print(vis_save_path)
        # print(error_mono[0], error_pred[0])
        # if error_mono[1]/error_pred[1]>1.3:
        # print(vis_save_path)
        view_idx(rgb_img, pred, vis_save_path)
        # view_per_img(rgb_img, pred, vis_save_path)