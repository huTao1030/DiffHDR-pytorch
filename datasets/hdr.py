import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
import glob
import cv2


class Hdr:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation=''):

        train_dataset = HdrDataset(n=self.config.training.patch_n,
                                  patch_size=self.config.data.image_size,
                                  transforms=self.transforms,
                                  training=True,
                                  parse_patches=parse_patches)
        val_dataset = HdrDataset(n=self.config.training.patch_n,
                                patch_size=self.config.data.image_size,
                                transforms=self.transforms,
                                training=False,
                                parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class HdrDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, training=True, parse_patches=True):
        super().__init__()

        self.dir = None
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.short_image_path = []
        self.medium_image_path = []
        self.long_image_path = []
        self.hdr_image_path = []
        self.exposure_path = []
        if training:
            self.data_path = glob.glob('./data/Training/0*')
        else:
            self.data_path = glob.glob('./data/Test/0*')

        for i in range(len(self.data_path)):
            self.short_image_path.append(self.data_path[i] + '/short.tif')
            self.medium_image_path.append(self.data_path[i] + '/medium.tif')
            self.long_image_path.append(self.data_path[i] + '/long.tif')
            self.hdr_image_path.append(self.data_path[i] + '/HDRImg.hdr')
            self.exposure_path.append(self.data_path[i] + '/exposure.txt')

    def imread_uint16_png(self,image_path):
        # Load image without changing bit depth
        return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    def mu_tonemap(self, hdr_image, mu=5000):
        return np.log(1 + mu * hdr_image) / np.log(1 + mu)

    def gamma_correction(self, img, expo, gamma):
        return (img ** gamma) / 2.0 ** expo

    def zero_padding(self, in_array):
        padding_array = np.zeros([1024,1536,6]).astype(np.float32)
        padding_array[0:1000, 0:1500] = in_array
        return padding_array

    def zero_padding_gt(self, in_array):
        padding_array = np.zeros([1024,1536,3]).astype(np.float32)
        padding_array[0:1000, 0:1500] = in_array
        return padding_array

    @staticmethod
    def get_params(img, output_size, n):
        w, h, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img[y[i]:y[i]+h, x[i]:x[i]+w]
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        # # input_name = self.medium_image_path[index]
        medium_image_name = self.medium_image_path[index]
        long_image_name = self.long_image_path[index]
        short_image_name = self.short_image_path[index]


        gt_name = self.hdr_image_path[index]
        img_id = re.split(r'\\', self.data_path[index])[-1]
        # input_img = cv2.cvtColor(cv2.imread(input_name), cv2.COLOR_BGR2RGB) / 255.0

        short_ldr = cv2.cvtColor(cv2.imread(short_image_name), cv2.COLOR_BGR2RGB) / 255.0
        medium_ldr = cv2.cvtColor(cv2.imread(medium_image_name), cv2.COLOR_BGR2RGB) / 255.0
        long_ldr = cv2.cvtColor(cv2.imread(long_image_name), cv2.COLOR_BGR2RGB) / 255.0
        label_hdr = self.imread_uint16_png(gt_name)

        s_gamma = 2.2
        exposure = []
        with open(self.exposure_path[index]) as lines:
            for line in lines:
                exposure.append(int(line[0]))

        medium_ldr_gamma = self.gamma_correction(medium_ldr, exposure[1], s_gamma)
        long_ldr_gamma = self.gamma_correction(long_ldr, exposure[2], s_gamma)
        short_ldr_gamma = self.gamma_correction(short_ldr, exposure[0], s_gamma)

        image_short_concat = np.concatenate((short_ldr, short_ldr_gamma), 2)
        image_medium_concat = np.concatenate((medium_ldr, medium_ldr_gamma), 2)
        image_long_concat = np.concatenate((long_ldr, long_ldr_gamma), 2)

        image_medium_concat = image_medium_concat.astype(np.float32)
        image_long_concat = image_long_concat.astype(np.float32)
        image_short_concat = image_short_concat.astype(np.float32)
        label_hdr = label_hdr.astype(np.float32)

        label_hdr = self.mu_tonemap(label_hdr)
        medium_img = image_medium_concat
        long_img = image_long_concat
        short_img = image_short_concat
        gt_img = label_hdr

        if self.parse_patches:
            i, j, h, w = self.get_params(medium_img, (self.patch_size, self.patch_size), self.n)

            medium_img = self.n_random_crops(medium_img, i, j, h, w)
            long_img = self.n_random_crops(long_img, i, j, h, w)
            short_img = self.n_random_crops(short_img, i, j, h, w)


            gt_img = self.n_random_crops(gt_img, i, j, h, w)
            outputs = [torch.cat([self.transforms(short_img[i]),self.transforms(medium_img[i]),self.transforms(long_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            short_img = self.zero_padding(short_img)
            medium_img = self.zero_padding(medium_img)
            long_img = self.zero_padding(long_img)
            gt_img = self.zero_padding_gt(gt_img)

            return torch.cat([self.transforms(short_img),self.transforms(medium_img),self.transforms(long_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.hdr_image_path)
