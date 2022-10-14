# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.backends.cudnn as cudnn
from DataSet import SceneDataset, ResnetEncoder
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

data_dirs = ['../Data/extracted_0_0', '../Data/extracted_0_1', '../Data/extracted_0_2']
if __name__ == '__main__':
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = ResnetEncoder()
    encoder.eval()

    classes = ('background', 'main', 'zoom')
    features = list()
    class_labels = list()
    image_labels = list()

    for j,  data_dir in enumerate(data_dirs):
        scene_dataset = SceneDataset(data_dir, transform=T.Resize(size=56))
        dataloader = torch.utils.data.DataLoader(scene_dataset, batch_size=1, shuffle=False, num_workers=1)

        for i, data in enumerate(dataloader, 0):
            if i % 4 == 0:
                # get the inputs
                inputs, labels = data
                if i % 25*4 == 0:
                    print(i, classes[labels[0]])
                img = inputs.float()
                features.append(img.reshape((16632,)))
                class_labels.append('f' + str(j) + '-' +str(int(labels[0].int())) + '-' + str(i))

                if labels[0] == 0:
                    image_labels.append(torch.tensor([[[0]], [[0]], [[0]]]))  # R
                elif labels[0] == 1:
                    image_labels.append(torch.tensor([[[0]], [[70*j]], [[70*j]]]))  # G
                elif labels[0] == 2:
                    image_labels.append(torch.tensor([[[70*j]], [[100]], [[45 + 70*j]]]))  # B

    writer = SummaryWriter('runs/input_1')
    features_torch = torch.stack(features)
    images_torch = torch.stack(image_labels)

    writer.add_embedding(features_torch, metadata=class_labels, label_img=images_torch)