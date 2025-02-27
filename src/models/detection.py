#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成模型代码（部分修改），支持动态调整 patch 提取步长实现正负样本再采样
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# ---------------------------
# PatchEmbedder：将 patch 打平后映射到低维表示
# ---------------------------
class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, embedding_dim, dm):
        """
        patch_size: 每个 patch 的长度（时间步数）
        embedding_dim: 时序数据每个时间步的维度（例如 256）
        dm: 输出 token 的维度（例如 64）
        """
        super(PatchEmbedder, self).__init__()
        self.input_dim = patch_size * embedding_dim
        self.linear = nn.Linear(self.input_dim, dm)

    def forward(self, patch):
        # patch: [patch_size, embedding_dim]
        flat = patch.view(-1)  # [patch_size * embedding_dim]
        token = self.linear(flat)  # [dm]
        return token

# ---------------------------
# PrototypeReprogrammer：利用多头注意力融合 learnable prototypes
# ---------------------------
class PrototypeReprogrammer(nn.Module):
    def __init__(self, dm, num_heads, V_prime, D, init_from_pretrained=False, pretrained_embeddings=None):
        """
        dm: 输入 patch token 的维度（例如 64）
        num_heads: 多头注意力头数
        V_prime: 原型数量（例如 1000）
        D: 目标表示空间维度，与预训练 BERT 隐藏层一致（例如 768）
        """
        super(PrototypeReprogrammer, self).__init__()
        self.dm = dm
        self.num_heads = num_heads
        if dm % num_heads != 0:
            raise ValueError("dm 必须能被 num_heads 整除")
        self.d = dm // num_heads
        self.D = D

        self.query_proj = nn.Linear(dm, dm)
        self.key_proj = nn.Linear(D, dm)
        self.value_proj = nn.Linear(D, dm)
        self.out_proj = nn.Linear(dm, D)

        if init_from_pretrained and pretrained_embeddings is not None:
            V_full = pretrained_embeddings.shape[0]
            indices = torch.randperm(V_full)[:V_prime]
            init_prototypes = pretrained_embeddings[indices].clone().detach()
            self.prototype_embeddings = nn.Parameter(init_prototypes)
        else:
            self.prototype_embeddings = nn.Parameter(torch.randn(V_prime, D))

    def forward(self, patch_tokens):
        """
        patch_tokens: [P, dm]，P 为 patch 数量
        输出: [P, D]
        """
        Q = self.query_proj(patch_tokens)  # [P, dm]
        prototypes = self.prototype_embeddings  # [V_prime, D]
        K = self.key_proj(prototypes)  # [V_prime, dm]
        V = self.value_proj(prototypes)  # [V_prime, dm]

        # 重塑为多头格式
        P = Q.size(0)
        Q = Q.view(P, self.num_heads, self.d).permute(1, 0, 2)  # [num_heads, P, d]
        V_prime = K.size(0)
        K = K.view(V_prime, self.num_heads, self.d).permute(1, 0, 2)  # [num_heads, V_prime, d]
        V = V.view(V_prime, self.num_heads, self.d).permute(1, 0, 2)  # [num_heads, V_prime, d]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d)  # [num_heads, P, V_prime]
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)  # [num_heads, P, d]
        attn_output = attn_output.permute(1, 0, 2).contiguous().view(P, self.dm)  # [P, dm]
        output = self.out_proj(attn_output)  # [P, D]
        return output

# ---------------------------
# PredictionHead：MLP 二分类预测层
# ---------------------------
class PredictionHead(nn.Module):
    def __init__(self, D, hidden_dim, output_dim=1):
        """
        D: 输入特征维度（例如 768）
        hidden_dim: MLP 隐藏层维度（例如 512）
        output_dim: 输出维度，二分类任务输出 1 个 logit
        """
        super(PredictionHead, self).__init__()
        self.fc1 = nn.Linear(D, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------------------------
# EndToEndModel：集成模型（融合时序 soft prompt 与聚合后的硬提示）
# ---------------------------
class EndToEndModel(nn.Module):
    def __init__(self, patch_size, emb_dim, dm, num_heads, V_prime, D, bert_model, prediction_head, stride=2):
        """
        patch_size: 每个 patch 长度
        emb_dim: 时序数据每步维度（例如 256）
        dm: PatchEmbedder 输出维度（例如 64）
        num_heads: 多头注意力头数
        V_prime: 原型数量
        D: 目标空间维度（例如 768，与 BERT 隐藏层一致）
        bert_model: 预加载的 BERT 模型（参数冻结）
        prediction_head: 用于二分类预测的 MLP 层
        stride: 默认滑动窗口步长
        """
        super(EndToEndModel, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.patch_embedder = PatchEmbedder(patch_size, emb_dim, dm)
        self.prototype_reprogrammer = PrototypeReprogrammer(dm, num_heads, V_prime, D)
        self.bert_model = bert_model
        self.prediction_head = prediction_head
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # 冻结 BERT 模型参数
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def extract_patches(self, sequence, stride=None):
        """
        sequence: [T, emb_dim]
        stride: 可选参数，如传入则覆盖模型初始化时定义的 stride，
           这样可以对正负样本用不同的 stride（例如正样本用较小 stride，负样本用较大 stride）
        返回： [num_patches, patch_size, emb_dim]
        """
        patches = []
        if stride is None:
            stride = self.stride
        T = sequence.size(0)
        for i in range(0, T - self.patch_size + 1, stride):
            patches.append(sequence[i:i+self.patch_size])
        if len(patches) == 0:
            patches.append(sequence[:self.patch_size])
        return torch.stack(patches, dim=0)

    def forward(self, sequence, aggregated_prompts, custom_stride=None):
        """
        sequence: [T, emb_dim] 时序数据
        aggregated_prompts: list of str, 长度应与 patch 数量一致，
           每个字符串为对应 patch 覆盖月份的聚合硬提示。
        custom_stride: 若不为 None，则用于替代默认的滑动窗口步长，实现不同标签的采样策略
        输出: 二分类 logit（标量）
        """
        # 1. 提取 patches（采用自定义 stride，以调整正负样本的采样密度）
        patches = self.extract_patches(sequence, stride=custom_stride)  # [num_patches, patch_size, emb_dim]
        num_patches = patches.size(0)
        patch_tokens = []
        for i in range(num_patches):
            token = self.patch_embedder(patches[i])
            patch_tokens.append(token)
        patch_tokens = torch.stack(patch_tokens, dim=0)  # [num_patches, dm]

        # 2. 得到 soft prompt tokens
        soft_prompts = self.prototype_reprogrammer(patch_tokens)  # [num_patches, D]

        # 3. 对每个 patch 的聚合硬提示进行编码（批量处理）
        inputs = self.tokenizer(aggregated_prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(sequence.device)  # [num_patches, L]
        hard_prompt_embeddings = self.bert_model.embeddings.word_embeddings(input_ids)  # [num_patches, L, D]

        # 4. 对每个 patch，将 soft prompt 作为首 token，与硬提示拼接
        soft_prompts_exp = soft_prompts.unsqueeze(1)  # [num_patches, 1, D]
        combined_embeddings = torch.cat([soft_prompts_exp, hard_prompt_embeddings], dim=1)  # [num_patches, L+1, D]

        # 5. 送入 BERT 模型
        outputs = self.bert_model(inputs_embeds=combined_embeddings)
        # outputs.last_hidden_state: [num_patches, L+1, D]
        # 对每个 patch 进行平均池化
        patch_embeddings = outputs.last_hidden_state.mean(dim=1)  # [num_patches, D]

        # 6. 全局汇聚（平均所有 patch 的表示）
        global_representation = patch_embeddings.mean(dim=0, keepdim=True)  # [1, D]
        prediction = self.prediction_head(global_representation)  # [1, 1]
        return prediction.squeeze()

# 后续消融模型（NoPrompt、NoReprogrammer）的改法类似，此处不再重复