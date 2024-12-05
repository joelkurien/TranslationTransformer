from typing import List
import torch
from torch import nn, Tensor, inf
from Linear import Linear
from ScaledDotProductAttention import ScaledDotProductAttention
from Operations import Operations

class MultiHeadAttention(nn.Module):
    def __init__(self, ipt: Tensor, h: int):
        super(MultiHeadAttention, self).__init__()
        self.ipt = ipt
        self.h = h
        self.Q = Tensor()
        self.K = Tensor()
        self.V = Tensor()
    
    def __init__(self, ipt: Tensor, opt: Tensor, h: int):
        super(MultiHeadAttention, self).__init__()
        self.ipt = ipt
        self.opt = opt
        self.h = h
        self.Q = Tensor()
        self.K = Tensor()
        self.V = Tensor()
        
    def set_QKV(self, weights):
        self.Q = Linear(self.ipt).forward(weights[0])
        self.K = Linear(self.ipt).forward(weights[1])
        self.V = Linear(self.ipt).forward(weights[2])
        
    def forward(self, weights: List[int], isMask: bool = False) -> Tensor:
        self.set_QKV(weights[:-1])
        _,d_q = self.Q.shape
        _,d_k = self.K.shape
        _,d_v = self.V.shape
        Q_head = self.Q.view(self.ipt.size(0), self.ipt.size(1), self.h, d_q)
        K_head = self.K.view(self.ipt.size(0), self.ipt.size(1), self.h, d_k)
        V_head = self.V.view(self.ipt.size(0), self.ipt.size(1), self.h, d_v)
        
        scale_attention_opt = ScaledDotProductAttention(Q_head, K_head, V_head).simple_attention(isMask)
        concat_opt = scale_attention_opt.transpose(1,2).contiguous().view(self.ipt.size(0), self.ipt.size(1), self.h * d_v)
        ma_opt = Linear(concat_opt).forward(weights[-1]) + self.ipt
        return Operations().layer_norm(ma_opt)