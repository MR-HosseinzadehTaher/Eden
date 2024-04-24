
import os
import torch
import random
import copy
from glob import glob
from PIL import Image
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import moco.loader
import math


class Transform:
    def __init__(self,crop_scale=(0.2, 1.)):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=crop_scale),
            transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        normalize
        ])
    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return y1, y2

class AnatomyDecomposerDataset(Dataset):
  def __init__(self, pathImageDirectory, pathDatasetFile, augment,n=2):
    self.img_list = []
    self.augment = augment
    self.n=n

    with open(pathDatasetFile, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
          imagePath = os.path.join(pathImageDirectory, lineItems[0])
          self.img_list.append(imagePath)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    if self.n ==0:
        x1, x2 = self.augment(imageData)
        return [x1,x2]
    else:
        image_array= np.array(imageData)
        patch_sx = np.random.randint(0, self.n)
        patch_sy = np.random.randint(0, self.n)
        patch_w = int(image_array.shape[0] // self.n)
        patch_h = int(image_array.shape[1] // self.n)
        patch = image_array[patch_sx*patch_w:(patch_sx+1)*patch_w, patch_sy*patch_h:(patch_sy+1)*patch_h, :]
        patch = Image.fromarray(patch)
        x1, x2 = self.augment(patch)
        return [x1, x2]

  def __len__(self):
    return len(self.img_list)

