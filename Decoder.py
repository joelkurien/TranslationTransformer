from typing import List
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork
from torch import nn, Tensor, inf
class Decoder: 
    
    def forward(self, values: Tensor, encoded_data: Tensor, weights: List[int], mask_weights: List[int]) -> Tensor:
        self_attn = MultiHeadAttention(values,8)
        self_attn_data = self_attn.forward(mask_weights, True)
        cross_attn = MultiHeadAttention(self_attn_data, 8, encoded_data)
        cross_attn_data = cross_attn.forward(weights)
        ffn = FeedForwardNetwork(cross_attn_data)
        return ffn.forward()