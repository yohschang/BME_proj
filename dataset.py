from unet_model import UNet
import torch.optim as optim
from tqdm import tqdm

import os
import random

import numpy as np
import torch
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import torch.nn as nn
from aslloss import AsymmetricLossOptimized
from scipy import ndimage


def normalize(x): return (x - np.min(x)) / \
    np.max(x - np.min(x)).astype(np.float32)


class MRIdata(Dataset):
    def __init__(self, onehot=False, Normal=True, transform=True):
        self.onehot = onehot
        self.normalize = Normal
        self.transform = transform
        self.datapath = [x for x in sorted(
            glob(r'tr/images/'+'*.nii.gz'), key=os.path.getmtime)]
        self.maskpath = [x for x in sorted(
            glob(r'tr/masks/'+'*.nii.gz'), key=os.path.getmtime)]

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):

        image = nib.load(self.datapath[idx])
        mask = nib.load(self.maskpath[idx])
        image_np = np.array(image.get_fdata()).astype(np.float32)
        mask_np = np.array(mask.get_fdata()).astype(np.float32)

        if self.normalize:
            image_np = normalize(image_np).astype(np.float32)

        if mask_np.shape[2] != 64:
            pad_size = 64 - mask_np.shape[2]
            pad_array = np.zeros(
                (image_np.shape[0], image_np.shape[1], pad_size//2), dtype=np.float32)
            if pad_size % 2 == 0:
                image_np = np.concatenate(
                    (pad_array, image_np, pad_array), axis=2)
                mask_np = np.concatenate(
                    (pad_array, mask_np, pad_array), axis=2)
            else:
                pad_array2 = np.zeros(
                    (image_np.shape[0], image_np.shape[1], pad_size//2+1), dtype=np.float32)
                image_np = np.concatenate(
                    (pad_array, image_np, pad_array2), axis=2)
                mask_np = np.concatenate(
                    (pad_array, mask_np, pad_array2), axis=2)

        if self.transform:
            # image = transpose(image_np)
            image_np = flip(image_np)
            image_np = rotate(image_np)

        image_t = transforms.ToTensor()(image_np)
        mask_t = transforms.ToTensor()(mask_np)

        one_hot_mask = torch.zeros((2,)+mask_np.shape)
        (one_hot_mask[0, :, :, :])[mask_t == 0] = 1
        (one_hot_mask[1, :, :, :])[mask_t == 1] = 1

        if self.onehot:
            return image_t, one_hot_mask

        else:
            return image_t, mask_t


class MRIslicedata(Dataset):
    def __init__(self, onehot=False, Normal=True):
        self.onehot = onehot
        self.normalize = Normal
        datapath = [x for x in sorted(
            glob(r'images/'+'*.nii.gz'), key=os.path.getmtime)]
        maskpath = [x for x in sorted(
            glob(r'masks/'+'*.nii.gz'), key=os.path.getmtime)]

        image_stack = [normalize(np.array(nib.load(img).get_fdata()))
                       for img in datapath]
        mask_stack = [normalize(np.array(nib.load(img).get_fdata()))
                      for img in maskpath]

        self.images = np.concatenate(image_stack, axis=2)
        self.masks = np.concatenate(mask_stack, axis=2)

    def __len__(self):
        return self.images.shape[2]

    def __getitem__(self, idx):

        image_np = self.images[:, :, idx].astype(np.float32)
        mask_np = self.masks[:, :, idx].astype(np.float32)

        image_t = transforms.ToTensor()(image_np)
        mask_t = transforms.ToTensor()(mask_np)

        if self.onehot:
            one_hot_mask = torch.zeros((2,)+image_np.shape)
            (one_hot_mask[0, :, :])[mask_t[0, ...] == 0] = 1
            (one_hot_mask[1, :, :])[mask_t[0, ...] == 1] = 1
            return image_t, one_hot_mask

        else:
            return image_t, mask_t


def flip(image):
    p1 = np.random.randint(2)
    if p1:
        image = image[::-1, :, :]
    p2 = np.random.randint(2)
    if p2:
        image = image[:, ::-1, :]

    return image


def rotate(image):
    angles = [90, 270, 180, 0]
    image = ndimage.rotate(image, np.random.choice(angles))
    return image


def transpose(image):
    p1 = np.random.randint(2)
    return [image, image.T][p1]


if __name__ == '__main__':

    trainset = MRIslicedata(onehot=True)
    trainset_Loader = DataLoader(trainset, batch_size=32, shuffle=True)
    image, mask = next(iter(trainset_Loader))

    print(image.size(), mask.size())
    print(len(trainset))
