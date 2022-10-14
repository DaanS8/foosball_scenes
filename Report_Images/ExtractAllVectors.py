from __future__ import print_function, division
import torch
import torch.backends.cudnn as cudnn
from DataSet import DirDataset, ResnetEncoder
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
import os

batch_size = 16

if __name__ == '__main__':
    with torch.no_grad():
        cudnn.benchmark = True
        rootdir = '..\Data'

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        encoder = ResnetEncoder(device)
        print('iscuda: ' + str(next(encoder.parameters()).is_cuda))
        encoder.eval()



        for file in os.listdir(rootdir):
            d = os.path.join(rootdir, file)
            if os.path.isdir(d):
                print(d)
                scene_dataset = DirDataset(d)
                print(len(scene_dataset))
                dataloader = torch.utils.data.DataLoader(scene_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

                classes = ('background', 'main', 'zoom')
                features = torch.zeros((len(scene_dataset), 512))
                for i, data in enumerate(dataloader):
                    # get the inputs
                    inputs, labels = data
                    if i % 5 == 0:
                        print(i, classes[labels[0]])
                    res = encoder(inputs.float())
                    features[batch_size*i: batch_size * i + res.shape[0]] = res.reshape((res.shape[0], 512))


                print(d + '.pt')
                torch.save(features, d + '.pt')
