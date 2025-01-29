# layers/Embed.py
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len=16, stride=8, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=patch_len, stride=stride)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        x = x.permute(0, 2, 1)  # (B, D, T)
        x = self.conv(x)  # (B, D, L)
        x = x.permute(0, 2, 1)  # (B, L, D)
        x = self.dropout(x)
        return x, x.size(1)