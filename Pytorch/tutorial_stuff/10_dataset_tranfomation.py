import torch 
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):

    def __init__(self, transform = None):
        xy = np.loadtxt('data/wine/wine.csv', delimiter=',', skiprows=1, dtype=np.float32)
        self.n_samples = xy.shape[0]

        self.x = xy[:,1:]
        self.y = xy[:,[0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class To_Tensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, taget = sample
        inputs *= self.factor
        return inputs, taget



dataset = WineDataset(transform=To_Tensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features)

composed = torchvision.transforms.Compose([To_Tensor(), MulTransform(3)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
