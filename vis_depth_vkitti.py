from PIL import Image
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def view_idx(rgb_img, pred, vis_save_path, error=None):
    plt.figure(figsize=(12,12))
    plt.subplot(4,1,1)
    plt.imshow(rgb_img)
    plt.title("tgt")

    plt.subplot(4,1,2)
    plt.imshow(1.0/pred[0], cmap = "plasma")
    plt.title("Monodepth2_"+str(error[0][0]))

    plt.subplot(4,1,3)
    plt.imshow(1.0/pred[1], cmap = "plasma")
    plt.title("Ours_"+str(error[1][0]))

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

def get_depth_gt(depth_png_filename):
    depth = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return depth

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
    rgb_dir = "dataset/VKitti"
    pred_dir = sys.argv[1]
    mono_dir = sys.argv[2]
    save_dir = sys.argv[3]
    make_dir(save_dir)

    MIN_DEPTH = 1e-3
    # MAX_DEPTH = 80
    MAX_DEPTH = 65535

    class_list = os.listdir(pred_dir)
    for cur_class in class_list:
        if 'tar' in cur_class:
            continue
        print(cur_class) # scene
        sub_dir = os.path.join(pred_dir, cur_class)
        type_list = os.listdir(sub_dir)
        for cur_type in type_list: # type_list
            if cur_type != '15-deg-left':
                continue
            
            sub_sub_dir = os.path.join(sub_dir, cur_type, 'Camera_0')
            input_file_list = os.listdir(sub_sub_dir)

            total_img = len(input_file_list)
            for i in tqdm(range(total_img-1)):
                try:
                    # load img pair
                    input_file = "rgb_"+str(i).zfill(5)+".npy"
                    
                    mono_path = os.path.join(mono_dir, cur_class, cur_type, 'Camera_0', input_file)
                    mono_disp = np.load(mono_path)[0][0]
                    pred_path = os.path.join(pred_dir, cur_class, cur_type, 'Camera_0', input_file)
                    pred_disp = np.load(pred_path)[0][0]

                    # gt_depth = get_depth_gt()
                    rgb_img = get_color(os.path.join(rgb_dir, cur_class, cur_type, 'frames', 'rgb', 'Camera_0', input_file.replace('npy', 'jpg')))
                    gt_depth = get_depth_gt(os.path.join(rgb_dir, cur_class, cur_type, 'frames', 'depth', 'Camera_0', input_file.replace('rgb', 'depth').replace('npy', 'png')))
                    
                    mono_depth, gt_depth_mask, mono_depth_mask = recale_depth(mono_disp, gt_depth)
                    error_mono = compute_errors(gt_depth_mask, mono_depth_mask)

                    pred_depth, gt_depth_mask, pred_depth_mask = recale_depth(pred_disp, gt_depth)
                    error_pred = compute_errors(gt_depth_mask, pred_depth_mask)

                    pred = [mono_depth, pred_depth]
                    error = [error_mono, error_pred]
                    # save_name = folder.split("/")[1] + "_" + str(frame_index) + "_" + side
                    save_name = cur_class + "_" + cur_type + "_" + str(i).zfill(5) + "_0"
                    vis_save_path = os.path.join(save_dir, save_name + ".jpg")
                    
                    view_idx(rgb_img, pred, vis_save_path, error)
                    # view_per_img(rgb_img, pred, vis_save_path)
                except Exception as e:
                    print(e)