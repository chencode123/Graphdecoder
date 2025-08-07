import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """
        A simple single-layer MLP with ReLU activation.

        :param in_dim: Dimension of input features
        :param out_dim: Dimension of output features
        """
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        :param x: Input tensor of shape [batch_size, in_dim] or [*, in_dim]
        :return: Output tensor of shape [batch_size, out_dim] or [*, out_dim]
        """
        return F.relu(self.linear(x))
