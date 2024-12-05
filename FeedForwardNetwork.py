import torch
from torch import nn, Tensor, inf
from Operations import Operations
class FeedForwardNetwork(nn.Module):
    def __init__(self, ipt: Tensor):
        super(FeedForwardNetwork, self).__init__()
        self.ipt = ipt
    
    def forward(self) -> Tensor:
        return Operations().layer_norm(nn.ReLU(self.ipt)+self.ipt)