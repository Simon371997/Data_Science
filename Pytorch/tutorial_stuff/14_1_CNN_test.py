import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper params
num_epochs = 0
batch_size = 4
learning_rate = 0.001

#dataset has PILImage image of range [0, 1]
#tranformation to normalized tensors range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5, 0.5, 0.5))]
)

train_dataset = torchvision.datasets.CIFAR10(root='./data2', train = True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data2', train = False, download=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

#get random training data
dataiter = iter(train_loader)
images, labels = next(dataiter)

#show images
#imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(3,6,5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6,16,5)
print(images.shape) #(4,3,32,32) batch size = 4, color channels = 3, size of pic = 32 x 32


x = conv1(images)
print(x.shape)  #(4,6,28,28) batch size = 4, output size of conv1 = 6,  convoluted size of pic = 28 x 28
x = pool(x)
print(x.shape) #(4,6,14,14) batch size = 4, output size of conv1 = 6,  reduced size of pic by factor 2 = 14 x 14
x = conv2(x)
print(x.shape) #(4,16,10,10) batch size = 4, output size of conv2 = 16,  convoluted size of pic = 10 x 10
x = pool(x)
print(x.shape) #(4,16,10,10) batch size = 4, output size of conv2 = 16,  reduced size of pic by factor 2 = 5 x 5