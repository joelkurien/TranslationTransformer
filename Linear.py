import torch
from torch import nn, Tensor, inf

class Linear(nn.Module):
    def __init__(self, ipt: Tensor):
        super(Linear, self).__init__()
        self.ipt = ipt
    
    def forward(self, weight: Tensor) -> Tensor:
        return torch.matmul(weight, self.ipt)