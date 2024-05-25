import glob
import os

import cv2
import numpy as np


def npz(im, la, re):
    images_path = im
    labels_path = la
    path2 = re
    images = os.listdir(images_path)
    for s in images:
        image_path = os.path.join(images_path, s)
        print(image_path)
        label_path = os.path.join(labels_path, s)
        # label_path = label_path.replace('jpg','png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 标签由三通道转换为单通道
        label = cv2.imread(label_path, flags=0)
        print(label_path)
        label[label != 255] = 0
        label[label == 255] = 1
        # 保存npz文件
        np.savez(path2 + s[:-4] + ".npz", image=image, label=label)


npz('/home/root1/wmz/GLD-water/train/image', '/home/root1/wmz/GLD-water/train/label', './datasets/GLH-Water/train_npz/')
npz('/home/root1/wmz/GLD-water/test/image', '/home/root1/wmz/GLD-water/test/label', './datasets/GLH-Water/test_vol_h5/')
