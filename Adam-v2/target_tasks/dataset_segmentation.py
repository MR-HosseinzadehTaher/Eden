from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import pydicom as dicom
import os
import cv2
from skimage import transform, io, img_as_float, exposure
import random
import copy
from einops import rearrange
import json
from torchvision import transforms
import albumentations
from albumentations import Compose, HorizontalFlip, Normalize, VerticalFlip, Rotate, Resize, ShiftScaleRotate, OneOf, GridDistortion, OpticalDistortion, \
    ElasticTransform, GaussNoise, MedianBlur,  Blur, CoarseDropout,RandomBrightnessContrast,RandomGamma,RandomSizedCrop, ToFloat

from albumentations.pytorch import ToTensorV2


def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;
    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]
    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height).T

#__________________________________________Pneumothorax Segmentation --------------------------------------------------
class PNEDataset(Dataset):
    def __init__(self, root,directory,transforms,dim=(224, 224, 3),normalization=None,annotation=100):
        self.directory = directory
        self.images_masks = np.load(self.directory, allow_pickle=True)
        self.transforms = transforms
        self.dim = dim
        self.root=root
        self.normalization = normalization
        if annotation < 100:
            _images_masks = copy.deepcopy(self.images_masks)
            self.images_masks = []
            healthy=[]
            diseased=[]
            for i in range (_images_masks.shape[0]):
                m=_images_masks[i]
                if m[1].replace(" ", "") == '-1':
                    healthy.append(m)
                else:
                    diseased.append(m)
            num_healthy = int(len(healthy) * annotation / 100)
            num_diseased = int(len(diseased) * annotation / 100)
            indexes_healthy = np.arange(len(healthy))
            indexes_diseased = np.arange(len(diseased))
            random.Random().shuffle(indexes_healthy)
            random.Random().shuffle(indexes_diseased)
            indexes_healthy = indexes_healthy[:num_healthy]
            indexes_diseased = indexes_diseased[:num_diseased]
            for i in indexes_healthy:
                self.images_masks.append(healthy[i])
            for j in indexes_diseased:
                self.images_masks.append(diseased[j])

            self.images_masks = np.array(self.images_masks)
        print("number of data:",len(self.images_masks))
    def __len__(self):
        return self.images_masks.shape[0]
    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_mask = self.images_masks[idx]
        ds = dicom.dcmread(os.path.join(self.root, str(image_mask[0])))
        img = np.array(ds.pixel_array)
        im = cv2.resize(img, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        im = (np.array(im)).astype('uint8')
        if len(im.shape) == 2:
            im = np.repeat(im[..., None], 3, 2)
        if image_mask[1].replace(" ", "") == '-1':
            msk = np.zeros((input_rows, input_cols), dtype="int")
        else:
            mask = rle2mask(image_mask[1], 1024, 1024)
            msk = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
            msk[msk > 0] = 255
            msk = (np.array(msk)).astype('uint8')
        if self.transforms:
                augmented = self.transforms(image=im, mask=msk)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(im) / 255.
            mask = np.array(msk) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0)
        return (im, mask)

#__________________________________________Thorax disease Segmentation, ChestX_Det dataset --------------------------------------------------
class ChestXDetDataset(Dataset):
    def __init__(self, pathImageDirectory, pathMaskDirectory, pathDatasetFile,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list = []
        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline().strip('\n')

                if line:
                    self.img_list.append(line)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            if num_data ==0:
                num_data =1
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.

        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std
        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)

#__________________________________________Lung, heart, clavicle Segmentation, SCR dataset --------------------------------------------------
class SCRDataset(Dataset):
    def __init__(self, pathImageDirectory, pathMaskDirectory, pathDatasetFile,transforms,dim=(224, 224, 3), anno_percent=100,num_class=1,normalization=None):
        self.transforms = transforms
        self.dim = dim
        self.pathImageDirectory=pathImageDirectory
        self.pathMaskDirectory =pathMaskDirectory
        self.normalization = normalization
        self.img_list = []
        with open(pathDatasetFile, "r") as fileDescriptor:
            line = True
            while line:
                line = fileDescriptor.readline().strip('\n')
                if line:
                    self.img_list.append(line)

        indexes = np.arange(len(self.img_list))
        if anno_percent < 100:
            random.Random(99).shuffle(indexes)
            num_data = int(indexes.shape[0] * anno_percent / 100.0)
            if num_data ==0:
                num_data =1
            indexes = indexes[:num_data]
            _img_list = copy.deepcopy(self.img_list)
            self.img_list = []
            for i in indexes:
                self.img_list.append(_img_list[i])
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        input_rows = self.dim[0]
        input_cols = self.dim[1]
        image_name= self.img_list[idx]
        image = Image.open(os.path.join(self.pathImageDirectory,image_name+".IMG.png"))
        image = image.convert('RGB')
        image = (np.array(image)).astype('uint8')
        mask = Image.open(os.path.join(self.pathMaskDirectory,image_name+".gif"))
        mask = mask.convert('L')
        mask = (np.array(mask)).astype('uint8')
        image = cv2.resize(image, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (input_rows, input_cols), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 255
        if self.transforms:
                augmented = self.transforms(image=image, mask=mask)
                im=augmented['image']
                mask=augmented['mask']
                im=np.array(im) / 255.
                mask=np.array(mask) / 255.
        else:
            im = np.array(image) / 255.
            mask = np.array(mask) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            im = (im-mean)/std

        mask = np.array(mask)
        im=im.transpose(2, 0, 1).astype('float32')
        mask=np.expand_dims(mask,axis=0).astype('uint8')
        return (im, mask)

#__________________________________________VinDR rib segmentation dataset --------------------------------------------------
class VinDrRibCXRDataset(Dataset):
    def __init__(self, image_path_file, image_size, mode,annotation=100):
        self.pathImageDirectory, pathDatasetFile = image_path_file
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        self.rib_labels =  ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10',
                           'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
        f = open(pathDatasetFile)
        data= json.load(f)
        self.img_list = data['img']
        self.label_list = data
        self.indexes = np.arange(len(self.img_list))
        if annotation < 100:
            random.Random().shuffle(self.indexes)
            num_data = int(self.indexes.shape[0] * annotation / 100.0)

            if num_data ==0:
                num_data =1
            self.indexes = self.indexes[:num_data]
        print("number of images:", len(self.indexes))

    def __getitem__(self, index):
        ind=self.indexes[index]
        imagePath = self.img_list[str(ind)]
        imageData = cv2.imread(os.path.join(self.pathImageDirectory, imagePath), cv2.IMREAD_COLOR)
        label0 = []
        for name in self.rib_labels:
            pts = self.label_list[name][str(ind)]
            label = np.zeros((imageData.shape[:2]), dtype=np.uint8)
            if pts != 'None':
                pts = np.array([[[int(pt['x']), int(pt['y'])]] for pt in pts])
                label = cv2.fillPoly(label, [pts], 1)
                label = cv2.resize(label, self.image_size,interpolation=cv2.INTER_AREA)
            label0.append(label)
        label0 = np.stack(label0)
        label0 = label0.transpose((1, 2, 0))

        imageData = cv2.resize(imageData,self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode] (image=imageData, mask=label0)
        img = dic['image']
        mask = (dic['mask'].permute(2, 0, 1))
        return img, mask
    def __len__(self):
        return len(self.indexes)