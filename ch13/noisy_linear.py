import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class NoisyLinear(nn.Module):
    """
    New dummy Layer that computes: w(x + ε) + b
    """
    def __init__(self, input_size, output_size, noise_stddev=0.1):
        super().__init__()
        w = torch.Tensor(input_size, output_size)
        self.w = nn.Parameter(w) # a Tensor
        nn.init.xavier_uniform_(self.w)
        b = torch.Tensor(output_size).fill_(0)
        self.b = nn.Parameter(b) # a Tensor
        self.noyse_stddev = noise_stddev

    def forward(self, x, training=False):
        if training:
            noise = torch.normal(0., self.noyse_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            x_new = x

        return torch.add(torch.mm(x_new, self.w), self.b)


torch.manual_seed(1)
noisy_layer = NoisyLinear(4, 2)
x = torch.zeros((1, 4))
for i in range(3):
    print(noisy_layer(x, training=True))

print(noisy_layer(x, training=False))