#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype Reprogrammer Module

该模块实现了利用预训练文本原型进行时序 patch reprogramming，
其核心思想来源于 TimeLLM 中的线性探测方法：
通过从预训练文本 embedding（E ∈ R^(V×D)）中线性探测出一个小型原型集合 E′ ∈ R^(V′×D)，
其中 V′ ≪ V，然后采用多头交叉注意力将输入的 patch token 与文本原型对齐，
生成与语言模型预训练空间对齐的表示。

设计说明：
  - 输入：patch token 序列，形状为 [P, dm]，dm 为 PatchEmbedder 的输出维度
  - 输出：reprogram 后的表示，形状为 [P, D]，D 为语言模型的隐藏维度
  - 操作流程：
      1. 对输入 token 进行线性映射得到 query（Q）
      2. 将 learnable 的文本原型 E′（初始化时可选依据预训练 embedding 初始值）分别映射为 key 和 value
      3. 将 Q、K、V 重塑为多头格式，计算 scaled dot-product attention（公式中每个 head k 的计算：
         Q(i)^k = patch_tokens × W_Q^k,  K(i)^k = E′ × W_K^k,  V(i)^k = E′ × W_V^k）
      4. 拼接所有 head 的输出后，再通过输出投影映射到目标空间 D

作者: Your Name
日期: 2025-02-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrototypeReprogrammer(nn.Module):
    def __init__(self, dm, num_heads, V_prime, D, init_from_pretrained=False, pretrained_embeddings=None):
        """
        参数:
          - dm: 输入 patch token 的维度（例如，PatchEmbedder 的输出维度）
          - num_heads: 多头注意力的头数，要求 dm 必须能被 num_heads 整除
          - V_prime: 文本原型数量，即 E′ 的行数
          - D: 目标特征空间的维度（与预训练语言模型一致）
          - init_from_pretrained: 如果为 True，则通过预训练文本 embedding 初始化文本原型
          - pretrained_embeddings: 如果 init_from_pretrained 为 True，需要传入预训练文本 embedding，
                                    形状为 [V, D]
        """
        super(PrototypeReprogrammer, self).__init__()
        self.dm = dm
        self.num_heads = num_heads
        if dm % num_heads != 0:
            raise ValueError("dm 必须能被 num_heads 整除")
        self.d = dm // num_heads  # 每个 head 的维度
        self.D = D

        # 对 patch token 进行线性映射生成 query
        self.query_proj = nn.Linear(dm, dm)

        # 将文本原型分别映射到 key 与 value 空间
        self.key_proj = nn.Linear(D, dm)
        self.value_proj = nn.Linear(D, dm)

        # 将多头注意力输出拼接后映射到目标维度 D
        self.out_proj = nn.Linear(dm, D)

        # 文本原型 E′，形状为 [V_prime, D]
        # 若 init_from_pretrained 为 True 且提供了 pretrained_embeddings，则从中采样初始化
        if init_from_pretrained and pretrained_embeddings is not None:
            V_full = pretrained_embeddings.shape[0]
            # 随机采样 V_prime 个 index 进行初始化
            indices = torch.randperm(V_full)[:V_prime]
            init_prototypes = pretrained_embeddings[indices].clone().detach()
            self.prototype_embeddings = nn.Parameter(init_prototypes)
        else:
            # 否则随机初始化文本原型
            self.prototype_embeddings = nn.Parameter(torch.randn(V_prime, D))

    def forward(self, patch_tokens):
        """
        输入：
          - patch_tokens: Tensor，形状 [P, dm]
        输出：
          - output: Tensor，形状 [P, D]
        """
        # 1. 对 patch token 生成 query，形状 [P, dm]
        Q = self.query_proj(patch_tokens)

        # 2. 文本原型，形状 [V_prime, D]
        prototypes = self.prototype_embeddings

        # 3. 将文本原型映射为 key 与 value，形状均为 [V_prime, dm]
        K = self.key_proj(prototypes)
        V = self.value_proj(prototypes)

        # 4. 重塑为多头格式
        # Q: [P, dm] -> [P, num_heads, d] -> [num_heads, P, d]
        P = Q.size(0)
        Q = Q.view(P, self.num_heads, self.d).permute(1, 0, 2)
        # K, V: [V_prime, dm] -> [V_prime, num_heads, d] -> [num_heads, V_prime, d]
        V_prime = K.size(0)
        K = K.view(V_prime, self.num_heads, self.d).permute(1, 0, 2)
        V = V.view(V_prime, self.num_heads, self.d).permute(1, 0, 2)

        # 5. 计算 scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d)  # [num_heads, P, V_prime]
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)  # [num_heads, P, d]

        # 6. 拼接各 head 的输出，并映射到目标维度
        attn_output = attn_output.permute(1, 0, 2).contiguous().view(P, self.dm)
        output = self.out_proj(attn_output)  # [P, D]
        return output


if __name__ == "__main__":
    torch.manual_seed(0)
    # 模拟10个 patch token，每个 token 维度为 dm=64
    patch_tokens = torch.randn(10, 64)
    model = PrototypeReprogrammer(dm=64, num_heads=4, V_prime=20, D=256, init_from_pretrained=False)
    output = model(patch_tokens)
    print("Output shape:", output.shape)  # 预期输出: [10, 256]