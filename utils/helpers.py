# utils/helpers.py

import torch
import torch.nn as nn
from math import sqrt


class AttentionFusion(nn.Module):
    """
    注意力融合模块，用于融合基础信息嵌入和重编程后的嵌入。
    """

    def __init__(self, embed_dim):
        """
        初始化注意力融合模块。

        参数：
            embed_dim (int): 嵌入向量的维度。
        """
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z, w):
        """
        前向传播。

        参数：
            z (torch.Tensor): 基础信息嵌入，形状为 (B, embed_dim)。
            w (torch.Tensor): 重编程后的嵌入，形状为 (V', embed_dim)。

        返回：
            out (torch.Tensor): 融合后的嵌入，形状为 (B, embed_dim)。
        """
        Q = self.query(z)  # (B, embed_dim)
        K = self.key(w)  # (V', embed_dim)
        V = self.value(w)  # (V', embed_dim)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(Q.size(-1))  # (B, V')
        attn = self.softmax(scores)  # (B, V')

        # 加权求和
        out = torch.matmul(attn, V)  # (B, embed_dim)
        return out


def load_prototypes(file_path):
    """
    加载预先聚类好的 LLM 词嵌入原型。

    参数：
        file_path (str): 原型文件的路径（.npy文件）。

    返回：
        W_prime (torch.Tensor): 原型嵌入矩阵，形状为 (V', D)。
    """
    W_prime = np.load(file_path)
    W_prime = torch.tensor(W_prime, dtype=torch.float)
    print(f"W_prime 加载完成，形状: {W_prime.shape}")
    return W_prime