# Source: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torchvision.models import resnet18


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


class ResnetAdapted(nn.Module):
    def __init__(self, device):
        self.device = device
        super(ResnetAdapted, self).__init__()
        resnet = resnet18().to(device)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        # dimension: (batchsize * n frame, 3, 227, 227)
        return self.resnet(images.to(self.device))
