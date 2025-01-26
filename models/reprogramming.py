import torch
import torch.nn as nn
from math import sqrt


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys, d_llm, dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.d_llm = d_llm

        self.query_proj = nn.Linear(d_model, n_heads * d_keys)
        self.key_proj = nn.Linear(d_llm, n_heads * d_keys)
        self.value_proj = nn.Linear(d_llm, n_heads * d_keys)
        self.out_proj = nn.Linear(n_heads * d_keys, d_llm)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        """
        Args:
            target_embedding: Tensor of shape (B, L, d_model)
            source_embedding: Tensor of shape (V', d_llm)
            value_embedding: Tensor of shape (V', d_llm)

        Returns:
            Tensor of shape (B, L, d_llm)
        """
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        K = self.d_keys

        # 投影
        Q = self.query_proj(target_embedding).view(B, L, H, K)  # (B, L, H, K)
        K_proj = self.key_proj(source_embedding).view(S, H, K)  # (S, H, K)
        V_proj = self.value_proj(value_embedding).view(S, H, K)  # (S, H, K)

        # 计算注意力分数
        scores = torch.einsum("blhk,shk->bhls", Q, K_proj) / sqrt(K)  # (B, H, L, S)

        # 应用 softmax
        attn = torch.softmax(scores, dim=-1)  # (B, H, L, S)
        attn = self.dropout(attn)

        # 加权求和
        out = torch.einsum("bhls,shk->blhk", attn, V_proj)  # (B, L, H, K)
        out = out.reshape(B, L, H * K)  # (B, L, H*K)

        # 最终投影
        return self.out_proj(out)  # (B, L, d_llm)