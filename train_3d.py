from unet_3d import Modified3DUNet
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
from dataset import MRIdata
from diceloss import DiceLoss


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


def pix_accuracy(pred, target):
    correct = torch.sum(pred == target).item()
    total   = torch.sum(target == target).item()
    return correct / total

def mean_IOU(pred, label):
    intersect = pred + label
    intersect_area = torch.sum(intersect != 0).item()
    cross_area = torch.sum(intersect == 2).item()
    
    if torch.sum(intersect == 2) == torch.sum(label == 1):
        iou = 1
    
    elif cross_area == 0 and intersect_area == 0:
        iou = 1
    else :
        iou = cross_area / intersect_area     
    return iou

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train():
  epoch = 50
  best_acc = 0
  
  for e in range(epoch):
    model.train()
    tot_loss = 0
    tot_acc = 0
    tot_iou = 0
    count = 0

    optimizer.param_groups[0]['lr'] = get_lr(optimizer) * np.power(0.985,e)

    print("lr : ", get_lr(optimizer) , ' | epoch : ' , e)

    for c , (images, masks) in enumerate(tqdm(trainset_Loader)):
      image, mask = images.to('cuda') ,masks.to('cuda') 

      image = image.unsqueeze(1)

      model.zero_grad()
      _, output = model(image)

      loss = critirion(output, mask)
      loss.backward()
      optimizer.step()
          
      tot_loss += loss.item()
      tot_acc += pix_accuracy(output.max(1)[1], mask.max(1)[1])
      tot_iou += mean_IOU(output.max(1)[1], mask.max(1)[1]) 
      count += 1


    print('=========================================')
    print('loss : ' + str(round(tot_loss/ count,4)))
    print('pix_acc : ' + str(round(tot_acc/ count,4)))
    print('mean IOU : ' + str(round(tot_iou/ count,4)))
    print('=========================================')

    val_acc = validation(model,valset_Loader,best_acc )
    if val_acc > best_acc:
      best_acc = val_acc
      save_path = model_name
      state = {"state_dict" : model.state_dict()}
      torch.save(state,save_path)



def validation(model, dataset,best_acc):
  model.eval()
  tot_acc = 0
  tot_iou = 0
  count = 0
  with torch.no_grad():
      for c , (images, masks) in enumerate(tqdm(dataset)):

        image, mask = images.to('cuda') ,masks.to('cuda') 
        image = image.unsqueeze(1)
        _, output = model(image)

        tot_acc += pix_accuracy(output.max(1)[1], mask.max(1)[1])
        tot_iou += mean_IOU(output.max(1)[1], mask.max(1)[1]) 
        count += 1

  print('================== validation =======================')
  print('pix_acc : ' + str(round(tot_acc/ count,4)))
  print('mean IOU : ' + str(round(tot_iou/ count,4)) + " / " + str(best_acc))
  print('================== validation =======================')

  return round(tot_iou/ count,4)


if __name__ == "__main__":

  model_name = "u_net_3dmodel_asl16.pt"

  dataset = MRIdata(onehot=True) 

  trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)])

  trainset_Loader = DataLoader(trainset, batch_size = 4 ,shuffle = True)
  valset_Loader = DataLoader(valset, batch_size =4, shuffle = True)

  model = Modified3DUNet(1,2,16).to("cuda")  # for 2d slice input
  if os.path.exists(model_name):
      checkpoint = torch.load(model_name)
      model.load_state_dict(checkpoint['state_dict'])
      print("model loaded")

#   critirion = nn.BCEWithLogitsLoss()
  critirion = AsymmetricLossOptimized()
  optimizer = optim.Adam(model.parameters(), lr = 0.01,weight_decay = 1e-5)

  train()


