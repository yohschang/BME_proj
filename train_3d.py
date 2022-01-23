'''
log_cosh_loss :https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
'''
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
from dataset import MRIdata

from tversky import FocalTversky_loss


class DiceLoss(nn.Module):

    def __init__(self, log_cosh=False):
        super(DiceLoss, self).__init__()
        self.log_cosh = log_cosh
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        dsc_loss = 1. - dsc

        if self.log_cosh:
            return torch.log((torch.exp(dsc_loss) + torch.exp(-dsc_loss))/2.0)

        else:
            return dsc_loss


def bce_dice_loss(y_true, y_pred):
    loss = nn.BCEWithLogitsLoss()(y_true, y_pred) + \
        DiceLoss()(y_true, y_pred)
    return loss / 2.0


def pix_accuracy(pred, target):
    pred = (pred > 0.5).float()  # onehot
    correct = torch.sum(pred == target).item()
    total = torch.sum(target == target).item()
    return correct / total


def mean_IOU(pred, label):
    pred = (pred > 0.5).float()  # onehot
    intersect = pred + label
    intersect_area = torch.sum(intersect != 0).item()
    cross_area = torch.sum(intersect == 2).item()

    if torch.sum(intersect == 2) == torch.sum(label == 1):
        iou = 1

    elif cross_area == 0 and intersect_area == 0:
        iou = 1
    else:
        iou = cross_area / intersect_area
    return iou


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train():
    epoch = 200
    best_acc = 0

    ori_lr = optimizer.param_groups[0]['lr']

    for e in range(epoch):
        model.train()
        tot_loss = 0
        tot_acc = 0
        tot_iou = 0
        count = 0

        optimizer.param_groups[0]['lr'] = ori_lr * np.power(0.95, e//2)

        print("lr : ", optimizer.param_groups[0]['lr'], ' | epoch : ', e)

        for c, (images, masks) in enumerate(tqdm(trainset_Loader)):
            image, mask = images.to('cuda'), masks.to('cuda')

            image = image.unsqueeze(1)
            mask = mask.unsqueeze(1)

            model.zero_grad()
            loss_cal, output = model(image)

            loss = critirion(output, mask)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            tot_acc += pix_accuracy(output, mask)
            tot_iou += mean_IOU(output, mask)
            count += 1

        print('=========================================')
        print('loss : ' + str(round(tot_loss / count, 4)))
        print('pix_acc : ' + str(round(tot_acc / count, 4)))
        print('mean IOU : ' + str(round(tot_iou / count, 4)))
        print('=========================================')

        val_acc = validation(model, valset_Loader, best_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = model_name
            state = {"state_dict": model.state_dict()}
            torch.save(state, save_path)


def validation(model, dataset, best_acc):
    model.eval()
    tot_acc = 0
    tot_iou = 0
    count = 0
    with torch.no_grad():
        for c, (images, masks) in enumerate(tqdm(dataset)):

            image, mask = images.to('cuda'), masks.to('cuda')
            image = image.unsqueeze(1)
            mask = mask.unsqueeze(1)
            _, output = model(image)

            tot_acc += pix_accuracy(output, mask)
            tot_iou += mean_IOU(output, mask)
            count += 1

    print('================== validation =======================')
    print('pix_acc : ' + str(round(tot_acc / count, 4)))
    print('mean IOU : ' + str(round(tot_iou / count, 4)) + " / " + str(best_acc))
    print('================== validation =======================')

    return round(tot_iou / count, 4)


if __name__ == "__main__":

    model_name = "u_net_3d_v2_bce"
    # model_name = "u_net_3d_bce_tronly"

    dataset = MRIdata(onehot=False, transform=False)

    trainset, valset = torch.utils.data.random_split(
        dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)])

    trainset_Loader = DataLoader(trainset, batch_size=8, shuffle=True)
    valset_Loader = DataLoader(valset, batch_size=8, shuffle=True)

    model = Modified3DUNet(1, 1).to("cuda")
    if os.path.exists(model_name):
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'])
        print("model loaded")

    # arg = {"apply_nonlin":None, "batch_dice":False, "do_bg":True, "smooth":1.,"square":False}
    # critirion = FocalTversky_loss(arg)
    critirion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    train()
