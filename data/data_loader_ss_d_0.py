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
        self.img_dir4 = os.path.join(img_folder, 'order4')
        self.img_names = [img_name for img_name in os.listdir(self.img_dir4) if os.path.isfile(os.path.join(self.img_dir4, img_name))]
        self.train = train

        if self.train:     # 只有在训练的时候才对图片进行shuffle
            random.shuffle(self.img_names)

        self.transform = transform
        assert len(self.img_names) > 0

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        assert index < len(self.img_names), 'index range error'
        img_path4 = os.path.join(self.img_dir4, self.img_names[index])
        img_path5 = img_path4.replace("order4", "order5")
        img_path6 = img_path4.replace("order4", "order6")
        img_path7 = img_path4.replace("order4", "order7")
        img_path8 = img_path4.replace("order4", "order8")
        img_path9 = img_path4.replace("order4", "order9")
        img4 = Image.open(img_path4).convert('RGB')
        img5 = Image.open(img_path5).convert('RGB')
        img6 = Image.open(img_path6).convert('RGB')
        img7 = Image.open(img_path7).convert('RGB')
        img8 = Image.open(img_path8).convert('RGB')
        img9 = Image.open(img_path9).convert('RGB')

        if self.transform is not None:
            img4 = self.transform(img4)
            img5 = self.transform(img5)
            img6 = self.transform(img6)
            img7 = self.transform(img7)
            img8 = self.transform(img8)
            img9 = self.transform(img9)

        imgs = [(1, img4), (1, img5), (1, img6), (0, img7), (0, img8), (0, img9)]

        if self.train:
            random.shuffle(imgs)

        return imgs, self.img_names[index]
