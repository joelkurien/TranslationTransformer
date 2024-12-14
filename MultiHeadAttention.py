from typing import List
import torch
from torch import nn, Tensor, inf
from Linear import Linear
from ScaledDotProductAttention import ScaledDotProductAttention
from Operations import Operations

class MultiHeadAttention(nn.Module):
    def __init__(self, ipt, h, opt= None):
        super().__init__()
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
        
    def forward(self, weights, isMask = False):
        self.set_QKV(weights[:-1])
        _,_,d_q = self.Q.shape
        _,_,d_k = self.K.shape
        _,_,d_v = self.V.shape
        
        batch_size, seq_len, d_model = self.ipt.size()  # Input size
        d_q = d_model // self.h  
        
        Q_head = self.Q.reshape(batch_size, seq_len, self.h, d_q).transpose(1, 2)  # [batch, heads, seq, d_q]
        K_head = self.K.reshape(batch_size, seq_len, self.h, d_q).transpose(1, 2)  # [batch, heads, seq, d_k]
        V_head = self.V.reshape(batch_size, seq_len, self.h, d_q).transpose(1, 2)  # [batch, heads, seq, d_v]

        scale_attention_opt = ScaledDotProductAttention(Q_head, K_head, V_head).simple_attention(isMask)
        concat_opt = scale_attention_opt.transpose(1,2).contiguous().view(batch_size, seq_len, d_model)
        ma_opt = Linear(concat_opt).forward(weights[-1]) + self.ipt
        return Operations().layer_norm(ma_opt)