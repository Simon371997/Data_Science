# Step function (not used in practice)
# Sigmoid (last layer of a classification problem)
# TanH (Hidden Layers) (scaled and shifted Sigmoid Function)
# ReLU (0 for Values < 0, value for values > 0)
# Leaky ReLU (slightly modified ReLU function)
# Softmax (good for last layer in multi-class-classification-problems)

import torch 
import torch.nn as nn
import torch.nn.functional as F

# option 1 (create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out 


# option 2 (use activation function directly in forward pass)
class NeuralNet2 (nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out 