# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.backends.cudnn as cudnn
from DataSet import SceneDataset, ResnetEncoder
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':
    cudnn.benchmark = True
    with torch.no_grad():

        data_dir = '../Data/extracted_0_2'
        scene_dataset = SceneDataset(data_dir)
        dataloader = torch.utils.data.DataLoader(scene_dataset, batch_size=1, shuffle=False, num_workers=1)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        encoder = ResnetEncoder(device)
        encoder.eval()

        classes = ('background', 'main', 'zoom')
        features = list()
        class_labels = list()
        image_labels = list()

        for i, data in enumerate(dataloader, 0):
            # get the inputs
            inputs, labels = data
            print(i, classes[labels[0]])
            img = inputs.float()
            features.append(encoder(img).reshape((512,)))
            class_labels.append(str(int(labels[0].int())) + '-' + str(i))

            if labels[0] == 0:
                image_labels.append(torch.tensor([[[255]], [[0]], [[0]]]))  # R
            elif labels[0] == 1:
                image_labels.append(torch.tensor([[[0]], [[255]], [[0]]]))  # G
            elif labels[0] == 2:
                image_labels.append(torch.tensor([[[0]], [[0]], [[255]]]))  # B

        writer = SummaryWriter('runs/fixed_0')
        features_torch = torch.stack(features)
        images_torch = torch.stack(image_labels)

        writer.add_embedding(features_torch, metadata=class_labels, label_img=images_torch)