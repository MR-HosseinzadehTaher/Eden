import torch
import numpy as np
import sys
from sklearn.metrics import roc_auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, optimizer, conf, epoch, save_file):
    print('==> Saving...',file=conf.log_writter)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def dice_score(im1, im2, empty_score=1.0):
    im1 = np.asarray(im1 > 0.5).astype(np.bool)
    im2 = np.asarray(im2 > 0.5).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum


def mean_dice_coef(y_true,y_pred):
    sum=0
    for i in range (y_true.shape[0]):
        sum += dice_score(y_true[i,:,:,:],y_pred[i,:,:,:])
    return sum/y_true.shape[0]

def torch_dice_coef_loss(y_true,y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

def iou(im1, im2):
    overlap = (im1 > 0.5) * (im2 > 0.5)
    union = (im1 > 0.5) + (im2 > 0.5)
    return overlap.sum() / float(union.sum())

def mean_iou(im1, im2):

    list = []
    for t in np.arange(0.5, 1.0, 0.05):
        overlap = (im1 >= t) * (im2 >= t)
        union = (im1 >= t) + (im2 >= t)
        fore_ground = overlap.sum() / float(union.sum())

        overlap = (im1 < t) * (im2 < t)
        union = (im1 < t) + (im2 < t)
        back_ground = overlap.sum() / float(union.sum())

        list.append((fore_ground+back_ground)/2)

    return np.mean(list)


def step_decay(step,conf):
    lr = conf.lr
    progress = (step - 20) / float(conf.epochs - 20)
    progress = np.clip(progress, 0.0, 1.0)
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))

    lr = lr * np.minimum(1., step / 20)

    return lr
