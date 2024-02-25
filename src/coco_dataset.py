import torch
import torchvision
from torchvision.datasets import VisionDataset
from pycocotools.coco import COCO
import os
import cv2
from copy import deepcopy
import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

def transform_image(train=False):
  if train:
    transform = A.Compose([
        A.Resize(640, 480, 3),
        # A.HorizontalFlip(p=0.3),
        # A.VerticalFlip(p=0.3),
        # A.RandomBrightnessContrast(p=0.1),
        # A.ColorJitter(p=0.1),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco'))
  else:
    transform = A.Compose([
        A.Resize(640, 480, 3),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco'))
  return transform

from typing import Any


class COCODataset(VisionDataset):
    def __init__(self, root, split='train', transforms=None, transform=None, target_transform=None):
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.json = os.path.join(root, 'annotations/person_keypoints_val2017.json')

        # Read annotations
        self.coco = COCO(self.json)
        self.split = split

        valid_ids = []
        for id in self.coco.getAnnIds():
            ann = self.coco.anns[id]
            if 1 in ann['keypoints'] or 2 in ann['keypoints']: # select only images where keypoints are present
                valid_ids.append(ann['image_id'])
        self.ids = valid_ids

        self.transforms = transforms

    def _load_image(self, id):
        path = os.path.join(self.root, 'images/val2017', self.coco.imgs[id]['file_name'])
        image = cv2.imread(path)

        return image
    
    def __getitem__(self, index: int):
        id = self.ids[index]
        image = self._load_image(id)
        # Load anns data
        all_keypoints = []
        keypoints = self.coco.loadAnns(self.coco.getAnnIds(id))
        for i, element in enumerate(keypoints):
            keypoint = element['keypoints']
            keypoint = np.array_split(keypoint, len(keypoint)/3) # chunk by three for (pixel x pixel x visibility)
            height, width = image.shape[:2] # take only pixels
            width_scale_factor = 640 / width
            height_scale_factor = 480 / height
            # scale pixels so that they match resized image
            for point in keypoint:
                point[0] = point[0] * width_scale_factor
                point[1] = point[1] * height_scale_factor
            all_keypoints.append(keypoint)
        image = cv2.resize(image, (640, 480)) # resize image

        torch_image = torchvision.transforms.ToTensor()(image)
        return torch_image.div(255.), image, all_keypoints

    # returns number of available images
    def __len__(self):
        return len(self.ids)
        




