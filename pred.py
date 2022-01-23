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
from dataset import MRIdata
from unet_3d import Modified3DUNet
from unet3d_v2.model import UNet3D


def validation(model, images, masks):
    model.eval()

    image, mask = images.to('cuda') ,masks.to('cuda') 
    image = image.unsqueeze(1)
    mask= mask.unsqueeze(1) #onehot
    # output = model(image)
    _,output = model(image)

    return output.cpu().detach().numpy()
    # output = model(image)



# model_name = "u_net_3d_v2_bce.pt"
model_name = "u_net_3dmodel_bcelogic_1dout.pt"

dataset = MRIdata(onehot=False, transform= False) 

dataset_Loader = DataLoader(dataset, batch_size = 1 ,shuffle = True)

image , mask = next(iter(dataset_Loader))

model = Modified3DUNet(1,1).to("cuda")  # for 2d slice input
# model = UNet3D(1,1).to("cuda")  # for 2d slice input u_net_3d_v2_bce.pt

if os.path.exists(model_name):
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint['state_dict'])
    print("model loaded")

out = validation(model,image, mask)

out[out < 0.5] = 0
out[out != 0] = 1
out = out[0,0,...]
mask = mask.numpy()[0,...]

# print(out.shape, mask.shape)

fig1, axs1 = plt.subplots(6,8,figsize=(15,15))
fig2, axs2 = plt.subplots(6,8,figsize=(15,15))

for i in range(8,64-8):
    i -= 8
    axs1[i//8, i%6].imshow(out[:,:,i])
    axs1[i//8, i%6].axis('off')
    axs2[i//8, i%6].imshow(mask[:,:,i])
    axs2[i//8, i%6].axis('off')
plt.show()


    # plt.imshow(out[:,:,i])
    # plt.axis("off")
    # plt.savefig("C:\\Users\\YX\\Desktop\\bmi_proj\\BME_proj\\result\\pred" +"\\" + str(i))
    # plt.imshow(mask[:,:,i])
    # plt.axis("off")
    # plt.savefig("C:\\Users\\YX\\Desktop\\bmi_proj\\BME_proj\\result\\mask" +"\\" + str(i))