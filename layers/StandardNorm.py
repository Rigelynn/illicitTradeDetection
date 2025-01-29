# layers/StandardNorm.py
import torch
import torch.nn as nn


class Normalize(nn.Module):
    def __init__(self, enc_in, affine=False):
        super(Normalize, self).__init__()
        self.norm = nn.LayerNorm(enc_in, elementwise_affine=affine)

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            return self.norm(x)
        elif mode == 'denorm':
            return self.norm(x)
        else:
            raise ValueError("Mode should be 'norm' or 'denorm'")