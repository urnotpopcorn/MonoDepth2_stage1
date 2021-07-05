import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.image as mpimg

pred_depth_path = "/Users/quiescence/Desktop/depth_all.npy"
pred_depths = np.load(pred_depth_path)

rgb_dir = "/Users/quiescence/Downloads/rgb/data"

def traverse_imgs(writer, pred_depths):
    for i in range(1, pred_depths.shape[0]):
        f_name = str(i).zfill(10)
        rgb_img = img = mpimg.imread(os.path.join(rgb_dir, f_name+".png"))

        plt.subplot(2,1,1)
        plt.imshow(rgb_img)
        plt.axis('off')

        plt.subplot(2,1,2)
        plt.imshow(1/pred_depths[i, :, :], cmap = "plasma")
        plt.axis('off')
        writer.grab_frame()
        # plt.pause(0.01)
        plt.clf()

        # plt.close()

metadata = dict(title='self-supervised-depth', artist='Matplotlib',comment='depth prediiton')
writer = FFMpegWriter(fps=10, metadata=metadata)

figure = plt.figure(figsize=(10.8, 7.2))
plt.ion()                                   
plt.tight_layout()
plt.subplots_adjust(wspace =0, hspace =0)
with writer.saving(figure, 'out.mp4', 100): 
    traverse_imgs(writer, pred_depths)