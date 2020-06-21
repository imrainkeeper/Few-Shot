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
    def __init__(self, img_folder, train=False, transform=None):
        self.img_dir1 = os.path.join(img_folder, str(0.32))
        self.img_names = [img_name for img_name in os.listdir(self.img_dir1) if os.path.isfile(os.path.join(self.img_dir1, img_name))]
        self.train = train

        if self.train:     # 只有在训练的时候才对图片进行shuffle
            random.shuffle(self.img_names)

        self.transform = transform
        assert len(self.img_names) > 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        assert index < len(self.img_names), 'index range error'
        img_path1 = os.path.join(self.img_dir1, self.img_names[index])
        img_path2 = img_path1.replace(str(0.32), str(0.4))
        img_path3 = img_path1.replace(str(0.32), str(0.5))
        img_path4 = img_path1.replace(str(0.32), str(0.625))
        img_path5 = img_path1.replace(str(0.32), str(0.78125))
        img1 = Image.open(img_path1).convert('RGB')
        img2 = Image.open(img_path2).convert('RGB')
        img3 = Image.open(img_path3).convert('RGB')
        img4 = Image.open(img_path4).convert('RGB')
        img5 = Image.open(img_path5).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)

        imgs = [(1, img1), (2, img2), (3, img3), (4, img4), (5, img5)]

        if self.train:
            random.shuffle(imgs)

        return imgs, self.img_names[index]
