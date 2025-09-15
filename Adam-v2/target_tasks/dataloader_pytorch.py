import os
import torch
import random
import copy
import csv
from glob import glob
from PIL import Image
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from skimage import measure
from skimage.transform import resize
import pydicom
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from PIL import ImageFilter
import pydicom as dicom

import cv2
import pandas as pd
import medpy.io

# ---------------------------------------------2D Data augmentation---------------------------------------------
class Augmentation():
  def __init__(self, normalize):
    if normalize.lower() == "imagenet":
      self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      self.normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() =="fundus":
      self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "none":
      self.normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)

  def get_augmentation(self, augment_name, mode, *args):
    try:
      aug = getattr(Augmentation, augment_name)
      return aug(self, mode, *args)
    except Exception as e:
      print (str(e))
      print("Augmentation [{}] does not exist!".format(augment_name))
      exit(-1)

  def basic(self, mode):
    transformList = []
    transformList.append(transforms.ToTensor())
    if self.normalize is not None:
      transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def _full(self, transCrop, transResize, mode="train", test_augment=True):
    transformList = []
    if mode == "train":
      transformList.append(transforms.RandomResizedCrop(transCrop))
      transformList.append(transforms.RandomHorizontalFlip())
      transformList.append(transforms.RandomRotation(7))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "valid":
      transformList.append(transforms.Resize((transResize,transResize)))
      transformList.append(transforms.CenterCrop(transCrop))
      transformList.append(transforms.ToTensor())
      if self.normalize is not None:
        transformList.append(self.normalize)
    elif mode == "test":
      if test_augment:
        transformList.append(transforms.Resize((transResize,transResize)))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if self.normalize is not None:
          transformList.append(transforms.Lambda(lambda crops: torch.stack([self.normalize(crop) for crop in crops])))
      else:
        transformList.append(transforms.Resize((transResize,transResize)))
        transformList.append(transforms.CenterCrop(transCrop))
        transformList.append(transforms.ToTensor())
        if self.normalize is not None:
          transformList.append(self.normalize)
    transformSequence = transforms.Compose(transformList)

    return transformSequence

  def full_224(self, mode, test_augment=True):
    transCrop = 224
    transResize = 256
    return self._full(transCrop, transResize, mode, test_augment=test_augment)

  def full_448(self, mode, test_augment=True):
    transCrop = 448
    transResize = 512
    return self._full(transCrop, transResize, mode, test_augment=test_augment)
  def full_1024(self, mode, test_augment=True):
    transCrop = 1024
    transResize = 1024
    return self._full(transCrop, transResize, mode, test_augment=test_augment)

  def full_896(self, mode, test_augment=True):
    transCrop = 896
    transResize = 1024
    return self._full(transCrop, transResize, mode, test_augment=test_augment)


class ChestX_ray14(Dataset):

  def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=14, anno_percent=100,in_channels=3):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.in_channels =in_channels
    with open(pathDatasetFile, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
          imagePath = os.path.join(pathImageDirectory, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if anno_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * anno_percent / 100.0)
      indexes = indexes[:num_data]
      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []
      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    if self.in_channels >1:
      imageData = Image.open(imagePath).convert('RGB')
    else:
      imageData = Image.open(imagePath).convert('L')
    imageLabel = torch.FloatTensor(self.img_label[index])
    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)

# ---------------------------------------------------ShenzenCXR DataSet------------------------------------------------#

class ShenzenCXR(Dataset):

  def __init__(self, pathImageDirectory, pathDatasetFile, augment, num_class=1, anno_percent=100,in_channels=3):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.in_channels = in_channels

    with open(pathDatasetFile, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')
          imagePath = os.path.join(pathImageDirectory, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)
    print("number of images:",len(self.img_list))

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    if self.in_channels >1:
      imageData = Image.open(imagePath).convert('RGB')
    else:
      imageData = Image.open(imagePath).convert('L')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)


# ---------------------------------------------Downstream VinDrCXR------------------------------------------
class VinDrCXR(Dataset):
    def __init__(self, images_path, file_path, augment, num_class=6, annotation_percent=100,in_channels=3):
        self.img_list = []
        self.img_label = []
        self.augment = augment
        self.in_channels = in_channels

        with open(file_path, "r") as fr:
            line = fr.readline().strip()
            while line:
                lineItems = line.split()
                imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                self.img_list.append(imagePath)
                self.img_label.append(imageLabel)
                line = fr.readline()

        if annotation_percent < 100:
            indexes = np.arange(len(self.img_list))
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * annotation_percent / 100.0)
            indexes = indexes[:num_data]

            _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
            self.img_list = []
            self.img_label = []

            for i in indexes:
                self.img_list.append(_img_list[i])
                self.img_label.append(_img_label[i])

    def __getitem__(self, index):

        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        if self.in_channels > 1:
          imageData = Image.open(imagePath).convert('RGB')
        else:
          imageData = Image.open(imagePath).convert('L')

        if self.augment != None: imageData = self.augment(imageData)
        return imageData, imageLabel
    def __len__(self):

        return len(self.img_list)

















