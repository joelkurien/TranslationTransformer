from typing import List
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork
from torch import nn, Tensor, inf
class Decoder: 
    
    def forward(self, values, encoded_data, weights, mask_weights):
        self_attn = MultiHeadAttention(values,8)
        self_attn_data = self_attn.forward(mask_weights, True)
        cross_attn = MultiHeadAttention(self_attn_data, 8, encoded_data)
        cross_attn_data = cross_attn.forward(weights)
        ffn = FeedForwardNetwork()
        return ffn.forward(cross_attn_data)