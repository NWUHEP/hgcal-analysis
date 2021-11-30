
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn import preprocessing 

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data      = torch.Tensor(data)
        self.targets   = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8).transpose(1,2,0))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()
        self.conv2d_1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.act_1    = nn.Relu()
        self.pool_1   = nn.MaxPool2d(2)
        self.conv2d_2 = nn.Conv2d(1, 8, kernel_size=2, padding=0, stride=2)
        self.act_2    = nn.Relu()
        self.pool_2   = nn.MaxPool2d(2)
        pooled_dim    = input_dim # n_conv_output / pooling_size
        self.encoder = nn.Sequential(
            nn.Linear(pooled_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(, 64),
            nn.ReLU()
            )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            #nn.Linear(256, 512),
            #nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.unsqueeze(1) # adds channel dimension for the case that there is one channel per image
        x = self.conv2d(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = x.view(-1, 8 * 8 * 8)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

