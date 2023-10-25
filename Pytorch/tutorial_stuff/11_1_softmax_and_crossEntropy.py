import torch
import torchvision
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis = 0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
#print('softmax numpy: ',outputs)

y = torch.tensor([2.0, 1.0, 0.1])
outputs_2 = torch.softmax(y, dim=0)
#print('softmax pytorch: ', outputs_2)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y muss be one hot encoded:
Y = np.array([1,0,0])

# y_pred has probabilities
y_pred_good = np.array([0.8, 0.15, 0.05])
y_pred_bad = np.array([0.2, 0.7, 0.1])
l1 = cross_entropy(Y, y_pred_good)
l2 = cross_entropy(Y, y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')



#Pytorch_version
loss = nn.CrossEntropyLoss()
Y2 = torch.tensor([2,0,1])


y2_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
y2_pred_bad = torch.tensor([[2.1, 1.0, 0.3], [0.1, 2.0, 21], [0.1, 3.0, 0.1 ]])

pl1 = loss(y2_pred_good, Y2)
pl2 = loss(y2_pred_bad, Y2)

print(f'Loss1 pytorch: {pl1.item():.4f}')
print(f'Loss2 pytorch: {pl2.item():.4f}')

_, predictions1 = torch.max(y2_pred_good,1)
_, predictions2 = torch.max(y2_pred_bad,1)
print(predictions1)
print(predictions2)
