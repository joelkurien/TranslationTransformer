from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork
from torch import nn, Tensor, inf
class Encoder: 
    
    def forward(self, query, weights):
        h = 8
        attn = MultiHeadAttention(query, h)
        ffn = FeedForwardNetwork()
        return ffn.forward(attn.forward(weights))