from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork
from torch import nn, Tensor, inf
class Encoder: 
    
    def forward(self, query: Tensor) -> Tensor:
        attn = MultiHeadAttention(query,8)
        ffn = FeedForwardNetwork()
        return ffn.forward(attn.forward(self.query))