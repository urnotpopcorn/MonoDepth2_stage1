import os
import sys
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def create_video(input_dir, rgb_input_dir, output_dir, curclass, crop_flag=False):
    sub_dir = os.path.join(input_dir, 'depth', curclass)
    input_path = os.path.join(sub_dir, str(0).zfill(5)+"_depth.jpg")
    depth = cv2.imread(input_path, 0)
    depth_height, depth_width = depth.shape
    size = (depth_width, depth_height*4)

    output_file = os.path.join(output_dir, curclass+".avi")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWrite = cv2.VideoWriter(output_file, fourcc, 3, size)

    rgb_sub_dir = os.path.join(rgb_input_dir, curclass)
    input_file_list = os.listdir(rgb_sub_dir)
    total_img = len(input_file_list) - 1
    for i in tqdm(range(total_img)):
        try:
            # rgb
            rgb_input_path = os.path.join(rgb_sub_dir, str(i).zfill(5)+".jpg")
            rgb = cv2.imread(rgb_input_path)

            if crop_flag == True:
                height, width, _ = rgb.shape
                w_h_ratio_cur = width * 1.0 / height
                w_h_ratio_tgt = depth_width * 1.0 / depth_height

                if w_h_ratio_cur >= w_h_ratio_tgt:
                    height_tgt = height
                    width_tgt = height * w_h_ratio_tgt

                    left = (width - width_tgt) // 2
                    top = 0
                    right = left + width_tgt
                    bottom = top + height_tgt
                else:
                    height_tgt = width / w_h_ratio_tgt
                    width_tgt = width

                    left = 0
                    top = (height - height_tgt) // 2
                    right = left + width_tgt
                    bottom = top + height_tgt
                    
                # rgb = rgb.crop((left, top, right, bottom))
                rgb = rgb[int(top): int(bottom), int(left): int(right)]

            rgb = cv2.resize(rgb, (depth_width, depth_height))

            # depth
            sub_dir = os.path.join(input_dir, 'depth', curclass)
            input_path = os.path.join(sub_dir, str(i).zfill(5)+"_depth.jpg")
            depth = cv2.imread(input_path)

            # flow
            sub_dir = os.path.join(input_dir, 'flow', curclass)
            input_path = os.path.join(sub_dir, str(i).zfill(5)+"_flow.jpg")
            flow = cv2.imread(input_path)
            
            # sceneflow
            sub_dir = os.path.join(input_dir, 'sceneflow', curclass)
            input_path = os.path.join(sub_dir, str(i).zfill(5)+"_sceneflow.jpg")
            sceneflow = cv2.imread(input_path)

            # mask
            sub_dir = os.path.join(input_dir, 'mask', curclass)
            input_path = os.path.join(sub_dir, str(i).zfill(5)+"_mask.jpg")
            mask = cv2.imread(input_path)

            img = np.vstack([rgb, depth, flow, mask])
            videoWrite.write(img)
        except Exception as e:
            print(e)
            continue
    videoWrite.release
    

def create_video_pertype(input_dir, output_dir, curtype, curclass):
    # type: depth/flow/sceneflow
    if curtype is None:
        sub_dir = os.path.join(input_dir, curclass)
        input_path = os.path.join(sub_dir, str(0).zfill(5)+".jpg")
        img = cv2.imread(input_path)  #读取第一张图片
        imgInfo = img.shape
        size = (imgInfo[1],imgInfo[0])  #获取图片宽高度信息
        output_file = os.path.join(output_dir, curclass+"_rgb.avi")
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWrite = cv2.VideoWriter(output_file, fourcc, 3, size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））

        input_file_list = os.listdir(sub_dir)
        total_img = len(input_file_list) - 1
        for i in tqdm(range(total_img)):
            try:
                input_path = os.path.join(sub_dir, str(i).zfill(5)+".jpg")
                img = cv2.imread(input_path)
                videoWrite.write(img)
            except Exception as e:
                print(e)
                continue
        videoWrite.release
    else:
        sub_dir = os.path.join(input_dir, curtype, curclass)
        input_path = os.path.join(sub_dir, str(0).zfill(5)+"_"+curtype+".jpg")
        img = cv2.imread(input_path)  #读取第一张图片
        imgInfo = img.shape
        size = (imgInfo[1],imgInfo[0])  #获取图片宽高度信息
        output_file = os.path.join(output_dir, curtype, curclass+"_"+curtype+".avi")
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWrite = cv2.VideoWriter(output_file, fourcc, 3, size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））

        input_file_list = os.listdir(sub_dir)

        if curtype == 'depth':
            total_img = len(input_file_list) - 1
        else:
            total_img = len(input_file_list)

        for i in tqdm(range(total_img)):
            try:
                input_path = os.path.join(sub_dir, str(i).zfill(5)+"_"+curtype+".jpg")
                img = cv2.imread(input_path)
                videoWrite.write(img)
            except Exception as e:
                print(e)
                continue
        videoWrite.release

def parse_input():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_dir", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--rgb_input_dir", type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_input()
    # model_dir = 'output/vis_video_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/stage123v2_monoori_woSIG_srcsample2_roi_y0/weights_11/'
    # rgb_input_dir = "/home/xzwu/xzwu/Code/GenerateSemantic/Dataset/DAVIS_2017/DAVIS-2019-Unsupervised-test-dev-Full-Resolution/JPEGImages/Full-Resolution"
    # depth_dir = os.path.join(args.input_dir, 'depth')
    # flow_dir = os.path.join(args.input_dir, 'flow')
    # sceneflow_dir = os.path.join(args.input_dir, 'sceneflow')
    
    # output/vis_video_DAVIS/DAVIS-2019-Unsupervised-test-dev-Full-Resolution_selected/stage1_monoori_woSIG_davis_bmx-bumps/weights_99/depth/
    class_list = os.listdir(os.path.join(args.input_dir, 'depth'))
    for cur_class in class_list:
        if 'sheep' in cur_class:
            continue
        print(cur_class)
        # create_video_pertype(args.input_dir, args.output_dir, curtype='depth', curclass=cur_class)
        # create_video_pertype(args.input_dir, args.output_dir, curtype='flow', curclass=cur_class)
        # create_video_pertype(args.input_dir, args.output_dir, curtype='sceneflow', curclass=cur_class)
        # create_video_pertype(args.rgb_input_dir, args.output_dir, curtype=None, curclass=cur_class)
        create_video(args.input_dir, args.rgb_input_dir, args.output_dir, cur_class, crop_flag=False)

main()