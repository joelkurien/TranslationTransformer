from typing import Tuple
import torch
from torch import nn, Tensor, inf

class Operations:
    def generate_self_weights_QKV(self, inpt_mat: Tensor, h: int = 8) -> Tuple[Tensor]:
        _,_,d_model = inpt_mat.shape
        Q_weight = nn.Parameter(torch.randn(d_model, d_model))
        K_weight = nn.Parameter(torch.randn(d_model, d_model))
        V_weight = nn.Parameter(torch.randn(d_model, d_model))
        O_weight = nn.Parameter(torch.randn(h*d_model, d_model))
        return Q_weight, K_weight, V_weight, O_weight

    def generate_cross_weights_QKV(self, inpt_mat: Tensor, otpt_mat: Tensor, h: int = 8) -> Tuple[Tensor]:
        _,_,d_ipt = inpt_mat.shape
        _,_,d_otpt = otpt_mat.shape
        Q_weight = nn.Parameter(torch.randn(d_ipt, d_ipt))
        K_weight = nn.Parameter(torch.randn(d_ipt, d_ipt))
        V_weight = nn.Parameter(torch.randn(d_otpt, d_otpt))
        O_weight = nn.Parameter(torch.randn(h*d_otpt, d_otpt))
        return Q_weight, K_weight, V_weight, O_weight

    def layer_norm(self, ipt: Tensor) -> Tensor:
        layer_norm = nn.LayerNorm(ipt.size()[1])
        return layer_norm(ipt)