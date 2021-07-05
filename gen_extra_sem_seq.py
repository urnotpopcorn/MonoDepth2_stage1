# I/O libraries
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, pretrain_model_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = tf.GraphDef.FromString(open(pretrain_model_path,'rb').read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        config=tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(graph=self.graph, config=config)

    def run(self, image, INPUT_TENSOR_NAME = 'ImageTensor:0', OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.
            INPUT_TENSOR_NAME: The name of input tensor, default to ImageTensor.
            OUTPUT_TENSOR_NAME: The name of output tensor, default to SemanticPredictions.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        # NOTE: use original image to predict
        batch_seg_map = self.sess.run(
            OUTPUT_TENSOR_NAME,
            feed_dict={INPUT_TENSOR_NAME: [np.asarray(image)]})
        seg_map = batch_seg_map[0]  # expected batch size = 1

        if len(seg_map.shape) == 2:
            seg_map = np.expand_dims(seg_map,-1)  # need an extra dimension for cv.resize
        
        # seg_map = cv.resize(seg_map, (width,height), interpolation=cv.INTER_NEAREST)
        
        return seg_map.astype(np.uint8)


def inference(raw_data_dir, kitti_data_sem_dir,
            date, seq, camera):

    # date = "2011_09_26"
    # seq = "2011_09_26_drive_0013_sync"
    # camera = "image_02"
    # input_dir = os.path.join(raw_data_dir, date, seq, camera, "data")
    # sem_dir = os.path.join(sem_dir, date, seq, camera, "data")
    # bbox_dir = os.path.join(bbox_dir, date, seq, camera, "data")
    # ins_dir = os.path.join(ins_dir, date, seq, camera, "data")
    # raw_data_dir = "dataset/raw_data"
    input_dir = os.path.join(raw_data_dir, date, seq, camera, "data")
    input_path_list = os.listdir(input_dir)
    input_len = len(input_path_list)
    for idx in tqdm(range(input_len)):
        # img_path: 2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000768.png
        # src_img_path: /local/xjqi/monodepth-project/kitti_data/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000768.png
        input_file = str(idx).zfill(10)
        src_img_path = os.path.join(input_dir, input_file+".png")
        rgb_img = Image.open(src_img_path).convert('RGB')

        # print("Predict: ", src_img_path)
        start_time = time.time()
        seg_map = MODEL.run(rgb_img)
        end_time = time.time()
        # print("inference time: ", end_time - start_time)

        dst_img_path = input_file+".npy"
        save_img_path = os.path.join(kitti_data_sem_dir, date, seq, camera, "data", dst_img_path)
        make_dir(os.path.dirname(save_img_path))
        np.save(save_img_path, seg_map)

def get_extra_lines(kitti_data_dir, kitti_data_ins_sem_dir, img_files, mode):
    # train_lines = get_extra_lines(kitti_data_dir, kitti_data_sem_dir, test_lines, mode="train")
    # 2011_09_30/2011_09_30_drive_0028_sync 2300 r

    img_path_files = []
    for img_path in img_files:
        frame_instant = int(img_path.split(" ")[1])
        t_frame_0 = str(frame_instant - 1)
        t_frame_1 = str(frame_instant)
        t_frame_2 = str(frame_instant + 1)

        img_index_0 = t_frame_0.zfill(10)
        img_index_1 = t_frame_1.zfill(10)
        img_index_2 = t_frame_2.zfill(10)

        if img_path.split(" ")[2] == "l":
            img_dir = "02"
        elif img_path.split(" ")[2] == "r":
            img_dir = "03"

        t_frame_0_src = os.path.join(kitti_data_ins_sem_dir, mode, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_0)+".npy")
        t_frame_1_src = os.path.join(kitti_data_ins_sem_dir, mode, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_1)+".npy")
        t_frame_2_src = os.path.join(kitti_data_ins_sem_dir, mode, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_2)+".npy")

        # print(t_frame_0_src)
        # print(t_frame_1_src)
        # print(t_frame_2_src)

        t_frame_0_dst = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_0)+".png")
        t_frame_1_dst = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_1)+".png")
        t_frame_2_dst = os.path.join(kitti_data_dir, img_path.split(" ")[0], "image_"+img_dir, "data", str(img_index_2)+".png")

        # print(t_frame_0_dst)
        # print(t_frame_1_dst)
        # print(t_frame_2_dst)

        if not os.path.exists(t_frame_0_src):
            if os.path.exists(t_frame_0_dst):
                img_path_files.append(img_path.split(" ")[0]+" "+str(img_index_0)+" "+img_path.split(" ")[2])

        if not os.path.exists(t_frame_1_src):
            if os.path.exists(t_frame_1_dst):
                img_path_files.append(img_path.split(" ")[0]+" "+str(img_index_1)+" "+img_path.split(" ")[2])

        if not os.path.exists(t_frame_2_src):
            if os.path.exists(t_frame_2_dst):
                img_path_files.append(img_path.split(" ")[0]+" "+str(img_index_2)+" "+img_path.split(" ")[2])

    return img_path_files


if __name__ == "__main__":
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    print("---------------------------------------------------------------")
    kitti_data_dir = os.path.join("dataset", "raw_data")
    kitti_data_sem_dir = os.path.join("dataset", "kitti_selected_mine", "sem")
    make_dir(kitti_data_sem_dir)

    pretrain_model_path ="deeplabv3_cityscapes_train/frozen_inference_graph.pb"
    MODEL = DeepLabModel(pretrain_model_path)
    
    print("inference...")
    date = "2011_09_26"
    camera = "image_02"

    # seq = "2011_09_26_drive_0013_sync"
    # inference(kitti_data_dir, kitti_data_sem_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0017_sync"
    # inference(kitti_data_dir, kitti_data_sem_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0018_sync"
    # inference(kitti_data_dir, kitti_data_sem_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0022_sync"
    # inference(kitti_data_dir, kitti_data_sem_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0051_sync"
    # inference(kitti_data_dir, kitti_data_sem_dir, date, seq, camera)

    # seq = "2011_09_26_drive_0005_sync"
    # inference(kitti_data_dir, kitti_data_sem_dir, date, seq, camera)

    seq = "2011_09_26_drive_0113_sync"
    inference(kitti_data_dir, kitti_data_sem_dir, date, seq, camera)
    
