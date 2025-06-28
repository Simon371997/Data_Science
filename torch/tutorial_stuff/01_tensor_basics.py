import torch
import numpy as np

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device = device)
    y = torch.ones()
    y = y.to(device)
    z = x+y

x = torch.ones(5, requires_grad=True)
print(x)