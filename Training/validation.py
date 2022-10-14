# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from DataSet import SceneDataset, ResnetAdapted
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    with torch.no_grad():
        print('Setup')
        plt.ion()   # interactive mode

        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.2, 0.22)
            ])
        }



        data_dir = '../Data/'
        image_datasets = {x: SceneDataset('../Data/val/', transform=data_transforms[x]) for x in ['val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}
        class_names = ['Background', 'Table', 'Zoom']

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['val']))

        # Load models and set to eval mode
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        model.load_state_dict(torch.load('../model_2.pt'))
        model_ft = model.to(device)
        model.eval()

        # start eval loop
        # First time through loop is slow
        print('GO!')
        min_conf = 1
        conf_sum = 0
        running_corrects = 0
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            conf = torch.nn.functional.normalize(torch.exp(outputs), dim=1)
            conf, preds = torch.max(conf, 1)  # argmax = predicted class

            min_conf = min(min_conf, torch.min(conf).float())
            conf_sum += torch.sum(conf)
            running_corrects += torch.sum(preds == labels.data)

        print('average confidence', str(float(conf_sum.double() / dataset_sizes['val'] * 100)) + '%')
        print('accuracy on validation set', str(float(running_corrects.double() / dataset_sizes['val'] * 100)) + '%')

