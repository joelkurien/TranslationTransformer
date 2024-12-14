import torch
from torch import nn, Tensor, inf

class Linear(nn.Module):
    def __init__(self, ipt):
        super(Linear, self).__init__()
        self.ipt = ipt
    
    def forward(self, weight):
        return torch.matmul(self.ipt, weight)