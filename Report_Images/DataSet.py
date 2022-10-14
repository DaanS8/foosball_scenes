# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from PIL import Image


class SceneDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(os.path.join(img_dir, 'labels.csv'))
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class DirDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len([name for name in os.listdir(self.img_dir) if (os.path.isfile(os.path.join(self.img_dir, name)) and name[-3:] == 'jpg')])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, 'frame_' + str(idx) + '.jpg')
        image = read_image(img_path)
        label = 1
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class ResnetEncoder(nn.Module):
    def __init__(self, model, device):
        self.device = device
        super(ResnetEncoder, self).__init__()
        resnet = model.to(device)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        # dimension: (batchsize * n frame, 3, 227, 227)
        return self.resnet(images.to(self.device)).cpu()
