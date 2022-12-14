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
    with open("out.txt", "a") as file_object:
        file_object.write('main\n')
    print('main')
    plt.ion()   # interactive mode

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.3, hue=.3),
            transforms.RandomRotation(degrees=(-20, 20)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.8),
            transforms.ToTensor(),
            transforms.Normalize(0.2, 0.22)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.3413, 0.2718)
        ]),
    }

    data_dir = '../Data/'
    image_datasets = {x: SceneDataset(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = ['Background', 'Table', 'Zoom']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(0.2)
        std = np.array(0.22)
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated
        plt.savefig('foo.png')


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])
    # store example of images
    with open("out.txt", "a") as file_object:
        file_object.write('GO!\n')

    print('GO!')
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            with open("out.txt", "a") as file_object:
                file_object.write(f'Epoch {epoch}/{num_epochs - 1}\n')
                file_object.write('-' * 10 + '\n')
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                with open("out.txt", "a") as file_object:
                    file_object.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), 'epoch_' + str(epoch) + '.model')
            with open("out.txt", "a") as file_object:
                file_object.write('\n')
            print()

        time_elapsed = time.time() - since
        with open("out.txt", "a") as file_object:
            file_object.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
            file_object.write(f'Best val Acc: {best_acc:4f}\n')
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 3)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=100)