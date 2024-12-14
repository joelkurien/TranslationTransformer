import math
import torch
from torch import nn, Tensor, inf

class ScaledDotProductAttention(nn.Module):
    def __init__(self, Q, K, V):
        super(ScaledDotProductAttention, self).__init__()
        self.Q = Q
        self.K = K
        self.V = V
    
    def simple_attention(self, isMask = False):
        d_k = self.K.shape[-1]
        QK = torch.matmul(self.Q, torch.transpose(self.K,-2,-1))
        scale = 1/math.sqrt(d_k)
        if isMask:
            mask = torch.triu(torch.ones_like(QK), diagonal=1).bool()
            QK = QK.masked_fill(mask, -inf)
        softmax = nn.Softmax(dim=1).forward(scale * QK)
        attention = torch.matmul(softmax, self.V)
        return attention    