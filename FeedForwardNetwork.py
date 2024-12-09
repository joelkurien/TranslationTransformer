import torch
from torch import nn, Tensor, inf
from Operations import Operations
class FeedForwardNetwork(nn.Module):
    def __init__(self):
        super(FeedForwardNetwork, self).__init__()
    
    def forward(self, ipt: Tensor) -> Tensor:
        return Operations().layer_norm(nn.ReLU(ipt)+ipt)