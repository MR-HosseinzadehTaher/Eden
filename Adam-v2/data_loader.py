
import os
import torch
import random
import copy
import csv
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import math
import cv2
import medpy.io

class CXRDataset(Dataset):
  def __init__(self, images_path, file_path, augment, data_granularity=0):
    self.img_list = []
    self.augment = augment
    self.data_granularity = data_granularity

    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
          imagePath = os.path.join(images_path, lineItems[0])
          self.img_list.append(imagePath)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    h, w, c = np.array(imageData).shape
    base_crop_size = int(min(h, w)// pow(2,self.data_granularity))
    return self.augment(imageData,base_crop_size)
  def __len__(self):
    return len(self.img_list)

class CXRDatasetHierarchical(Dataset):
  def __init__(self, images_path, file_path, augment, data_granularity=2, nb_parts=4):
    self.img_list = []
    self.augment = augment
    self.data_granularity = data_granularity
    self.nb_parts = nb_parts
    # number of parts should be power of 4
    assert self.nb_parts%4==0

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()
          imagePath = os.path.join(images_path, lineItems[0])
          self.img_list.append(imagePath)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    image_array = np.array(imageData)
    inputs = []
    #whole
    if self.data_granularity ==0:
        inputs.append(imageData)
    else:
        # random crop
        patch_w = int(image_array.shape[0] // pow(2,self.data_granularity))
        patch_h = int(image_array.shape[1] // pow(2,self.data_granularity))
        if (image_array.shape[0] - patch_w - 50) > 50:
            patch_sx = np.random.randint(50, image_array.shape[0] - patch_w - 50)
        else:
            patch_sx = np.random.randint(0, image_array.shape[0] - patch_w -1 )

        if (image_array.shape[1] - patch_h - 50) > 50:
            patch_sy = np.random.randint(50, image_array.shape[1] - patch_h - 50)
        else:
            patch_sy = np.random.randint(0, image_array.shape[1] - patch_h - 1)
        image_array = image_array[patch_sx:patch_sx+patch_w, patch_sy:patch_sy+patch_h,:]
        inputs.append(Image.fromarray(image_array))
    # parts
    part_w = int(image_array.shape[0] // math.sqrt(self.nb_parts))
    part_h = int(image_array.shape[1] // math.sqrt(self.nb_parts))
    for i in range(int(math.sqrt(self.nb_parts))):
        for j in range(int(math.sqrt(self.nb_parts))):
            part = image_array[i * part_w:(i + 1) * part_w, j * part_h:(j + 1) * part_h, :]
            part = Image.fromarray(part)
            inputs.append(part)
    return self.augment(inputs)
  def __len__(self):
    return len(self.img_list)

