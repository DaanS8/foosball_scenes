# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.backends.cudnn as cudnn
from DataSet import SceneDataset, ResnetEncoder
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter

data_dirs = ['../Data/extracted_0_0', '../Data/extracted_0_1', '../Data/extracted_0_2', '../Data/extracted_0_3']
if __name__ == '__main__':
    with torch.no_grad():
        cudnn.benchmark = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        encoder = ResnetEncoder(device)
        encoder.eval()

        classes = ('background', 'main', 'zoom')
        features = list()
        class_labels = list()
        image_labels = list()

        colors = [[[[[66]], [[117]], [[245]]], [[[172]], [[223]], [[227]]], [[[0]], [[132]], [[255]]], [[[164]], [[89]], [[240]]]],
                  [[[[255]], [[0]], [[0]]], [[[201]], [[74]], [[48]]], [[[255]], [[141]], [[92]]], [[[255]], [[123]], [[0]]]]]
        for j,  data_dir in enumerate(data_dirs):
            scene_dataset = SceneDataset(data_dir)
            dataloader = torch.utils.data.DataLoader(scene_dataset, batch_size=1, shuffle=False, num_workers=1)

            for i, data in enumerate(dataloader, 0):
                # get the inputs
                inputs, labels = data
                if i%25 == 0:
                    print(i, classes[labels[0]])
                img = inputs.float()
                features.append(encoder(img).reshape((512,)))
                class_labels.append("f" + str(j) + "-" + str(int(labels[0].int())) + '-' + str(i))

                if labels[0] == 0:
                    image_labels.append(torch.tensor([[[0]], [[0]], [[0]]]))
                elif labels[0] == 1:
                    image_labels.append(torch.tensor(colors[0][j]))
                elif labels[0] == 2:
                    image_labels.append(torch.tensor(colors[1][j]))

        writer = SummaryWriter('runs/fixed_1')
        features_torch = torch.stack(features)
        images_torch = torch.stack(image_labels)

        writer.add_embedding(features_torch, metadata=class_labels, label_img=images_torch)