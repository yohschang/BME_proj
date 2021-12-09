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


class MRIdata(Dataset):
  def __init__(self, onehot = False):
    self.onehot = onehot
    self.datapath = [x for x in sorted(glob(r'images/'+'*.nii.gz'),key=os.path.getmtime)]
    self.maskpath = [x for x in sorted(glob(r'masks/'+'*.nii.gz'),key=os.path.getmtime)]

  def __len__(self):
    return len(self.datapath)

  def __getitem__(self,idx):

    image = nib.load(self.datapath[idx])
    mask = nib.load(self.maskpath[idx])
    image_np = np.array(image.get_fdata()).astype(np.float32)
    mask_np = np.array(mask.get_fdata()).astype(np.float32)

    if mask_np.shape[2] != 64:
        pad_size = 64- mask_np.shape[2] 
        pad_array = np.zeros((image_np.shape[0], image_np.shape[1],pad_size//2),dtype = np.float32)
        if pad_size%2 ==0:
            image_np = np.concatenate((pad_array,image_np,pad_array), axis=2)
            mask_np = np.concatenate((pad_array,mask_np,pad_array), axis=2)
        else:
            pad_array2 = np.zeros((image_np.shape[0], image_np.shape[1],pad_size//2+1),dtype = np.float32)
            image_np = np.concatenate((pad_array,image_np,pad_array2), axis=2)
            mask_np = np.concatenate((pad_array,mask_np,pad_array2), axis=2)



    image_t = transforms.ToTensor()(image_np)
    mask_t = transforms.ToTensor()(mask_np)

    one_hot_mask = torch.zeros((2,)+mask_np.shape)
    (one_hot_mask[0,:,:,:])[mask_t == 0] = 1
    (one_hot_mask[1,:,:,:])[mask_t == 1] = 1
    
    if self.onehot:
        return image_t, one_hot_mask
    
    else:
        return image_t , mask_t

if __name__ == '__main__':

    trainset = MRIdata() 
    trainset_Loader = DataLoader(trainset, batch_size = 1, shuffle = True)
    # image , mask = next(iter(trainset_Loader))


    for img, mask, pth in trainset_Loader:

        # print("++++++++++++++++++++++++++++++++++++")
        # print(img.size())
        # print(mask.size())
        # print(pth)
        if img.size()[1] != 64:
            print('not 64 ==========')
            print(img.size())
            print(mask.size())
            print(pth)