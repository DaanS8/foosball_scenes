# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division
import torch
import torch.backends.cudnn as cudnn
from DataSet import SceneDataset, ResnetEncoder
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

all_loops = [['../Data/extracted_0_2'], ['../Data/extracted_0_0', '../Data/extracted_0_1', '../Data/extracted_0_2'], ["../Data/train"], ["../Data/val"]]

names = ['single', 'multiple', 'train', 'val']

if __name__ == '__main__':
    with torch.no_grad():
        cudnn.benchmark = True
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        defaultEncoder = ResnetEncoder(models.resnet18(), device)
        defaultEncoder.eval()

        # Use resnet 18 model, but with only 3 classes
        model = models.resnet18()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
        # Loading (best) trained model
        model.load_state_dict(torch.load('model_1.pt'))

        trained_encoder = ResnetEncoder(model, device)
        trained_encoder.eval()

        classes = ('background', 'main', 'zoom')

        # assumes images have correct size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.3413, 0.2718)
        ])

        transform_input = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor()
        ])

        for z, cur_dirs in enumerate(all_loops):
            print('Directories:', z)
            features_input = list()
            features_default = list()
            features_trained = list()
            class_labels = list()

            for j,  data_dir in enumerate(cur_dirs):
                input_dataset = SceneDataset(data_dir, transform=transform_input)
                scene_dataset = SceneDataset(data_dir, transform=transform)
                dataloader = torch.utils.data.DataLoader(scene_dataset, batch_size=1, shuffle=False, num_workers=1)

                for i, data in enumerate(input_dataset):
                    features_input.append(torch.flatten(data[0]))

                for i, data in enumerate(dataloader):
                    # get the inputs
                    inputs, labels = data
                    class_labels.append(labels[0])
                    features_default.append(defaultEncoder(inputs)[0].squeeze())
                    features_trained.append(trained_encoder(inputs)[0].squeeze())

            print('Finished extracting')
            features_input = torch.stack(features_input)
            features_default = torch.stack(features_default)
            features_trained = torch.stack(features_trained)
            class_labels = torch.stack(class_labels)

            print('PCA input figure')
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(features_input)

            ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
            ax.scatter(
                xs=pca_result[:, 0],
                ys=pca_result[:, 1],
                zs=pca_result[:, 2],
                c=class_labels,
                cmap='tab10'
            )
            ax.set_xlabel('pca-one')
            ax.set_ylabel('pca-two')
            ax.set_zlabel('pca-three')
            plt.savefig(names[z] + '_input.png')

            print('PCA default figure')
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(features_default)
            plt.show()

            ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
            ax.scatter(
                xs=pca_result[:,0],
                ys=pca_result[:,1],
                zs=pca_result[:,2],
                c=class_labels,
                cmap='tab10'
            )
            ax.set_xlabel('pca-one')
            ax.set_ylabel('pca-two')
            ax.set_zlabel('pca-three')
            plt.savefig(names[z] + '_default.png')
            plt.show()

            print('PCA trained figure')
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(features_trained)

            ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
            ax.scatter(
                xs=pca_result[:, 0],
                ys=pca_result[:, 1],
                zs=pca_result[:, 2],
                c=class_labels,
                cmap='tab10'
            )
            ax.set_xlabel('pca-one')
            ax.set_ylabel('pca-two')
            ax.set_zlabel('pca-three')
            plt.savefig(names[z] + '_trained.png')
            plt.show()


