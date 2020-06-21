############################################
#
############################################

import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import h5py
import cv2


class ImageDataset(Dataset):
    def __init__(self, img_folder_cc, img_folder_d, train=False, transform=None):
        self.img_folder_d = img_folder_d
        self.img_folder_cc = img_folder_cc

        # 每次输入网络的都是从同一张图片中crop出来的5个crowd count图片块以及6个distance图片块
        self.img_names = [img_name for img_name in os.listdir(os.path.join(img_folder_d, 'order4')) if os.path.isfile(os.path.join(img_folder_d, 'order4', img_name))]

        self.train = train

        if self.train:     # 只有在训练的时候才对图片进行shuffle
            random.shuffle(self.img_names)

        self.transform = transform
        assert len(self.img_names) > 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        assert index < len(self.img_names), 'index range error'
        img_path_d_4 = os.path.join(self.img_folder_d, 'order4', self.img_names[index])
        img_path_d_5 = img_path_d_4.replace("order4", "order5")
        img_path_d_6 = img_path_d_4.replace("order4", "order6")
        img_path_d_7 = img_path_d_4.replace("order4", "order7")
        img_path_d_8 = img_path_d_4.replace("order4", "order8")
        img_path_d_9 = img_path_d_4.replace("order4", "order9")
        img1 = Image.open(img_path_d_4).convert('RGB')
        img2 = Image.open(img_path_d_5).convert('RGB')
        img3 = Image.open(img_path_d_6).convert('RGB')
        img4 = Image.open(img_path_d_7).convert('RGB')
        img5 = Image.open(img_path_d_8).convert('RGB')
        img6 = Image.open(img_path_d_9).convert('RGB')

        img_path_cc_1 = os.path.join(self.img_folder_cc, "0.32", self.img_names[index])[:-11] + '_' + str(0.32) + '.jpg'
        img_path_cc_2 = img_path_cc_1.replace(str(0.32), str(0.4))
        img_path_cc_3 = img_path_cc_1.replace(str(0.32), str(0.5))
        img_path_cc_4 = img_path_cc_1.replace(str(0.32), str(0.625))
        img_path_cc_5 = img_path_cc_1.replace(str(0.32), str(0.78125))
        img7 = Image.open(img_path_cc_1).convert('RGB')
        img8 = Image.open(img_path_cc_2).convert('RGB')
        img9 = Image.open(img_path_cc_3).convert('RGB')
        img10 = Image.open(img_path_cc_4).convert('RGB')
        img11 = Image.open(img_path_cc_5).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)
            img6 = self.transform(img6)
            img7 = self.transform(img7)
            img8 = self.transform(img8)
            img9 = self.transform(img9)
            img10 = self.transform(img10)
            img11 = self.transform(img11)

        imgs = [(0, img1), (0, img2), (0, img3), (1, img4), (1, img5), (1, img6), (10, img7), (11, img8), (12,img9), (13, img10), (14, img11)]

        if self.train:
            random.shuffle(imgs)

        return imgs, self.img_names[index]
