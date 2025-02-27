#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成模型代码，包括：
    - PatchEmbedder：将时序数据 patch 打平后映射到低维表示
    - PrototypeReprogrammer：利用多头注意力融合 learnable prototypes 生成软提示（soft prompt）
    - PredictionHead：MLP 二分类预测层
    - EndToEndModel：集成模型，融合时序 soft prompt 与聚合后的硬提示（hard prompt）
    - 消融实验变体：EndToEndModel_NoPrompt 和 EndToEndModel_NoReprogrammer

本文件修改自原 LLaMA 版本，目前采用预训练的 BERT 模型（例如 bert-base-uncased），因此注意 BERT 的隐藏层维度通常为 768。
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
        self.patch_size = patch_size
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

    def extract_patches(self, sequence, custom_stride=None):
        """
        sequence: [T, emb_dim]
        custom_stride: 可选参数，若不为 None 则使用自定义步长，否则使用默认 self.stride
        返回： [num_patches, patch_size, emb_dim]
        """
        patches = []
        stride = custom_stride if custom_stride is not None else self.stride
        T = sequence.size(0)
        for i in range(0, T - self.patch_size + 1, stride):
            patches.append(sequence[i:i + self.patch_size])
        if len(patches) == 0:
            patches.append(sequence[:self.patch_size])
        return torch.stack(patches, dim=0)

    def forward(self, sequence, aggregated_prompts, custom_stride=None):
        """
        sequence: [T, emb_dim] 时序数据
        aggregated_prompts: list of str，原本长度应与默认 stride 下生成的 patch 数一致；
           若使用自定义 stride 后两者数量不匹配，则对 aggregated_prompts 进行简单调整（截断或补齐）。
        custom_stride: 若不为 None，则用于替代默认步长，实现针对不同标签的采样策略
        输出: 二分类 logit（标量）
        """
        # 1. 提取 patches，动态调整采样粒度
        patches = self.extract_patches(sequence, custom_stride)
        num_patches = patches.size(0)

        # 若 aggregated_prompts 数量与生成的 patch 数不一致，则补齐或截断
        if len(aggregated_prompts) != num_patches:
            if len(aggregated_prompts) < num_patches:
                aggregated_prompts = aggregated_prompts + [aggregated_prompts[-1]] * (num_patches - len(aggregated_prompts))
            else:
                aggregated_prompts = aggregated_prompts[:num_patches]

        # 2. 对每个 patch 生成 token
        patch_tokens = []
        for i in range(num_patches):
            token = self.patch_embedder(patches[i])
            patch_tokens.append(token)
        patch_tokens = torch.stack(patch_tokens, dim=0)  # [num_patches, dm]

        # 3. 得到 soft prompt tokens
        soft_prompts = self.prototype_reprogrammer(patch_tokens)  # [num_patches, D]

        # 4. 对每个 patch 的聚合硬提示进行编码（批量处理）
        inputs = self.tokenizer(aggregated_prompts, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(sequence.device)  # [num_patches, L]
        hard_prompt_embeddings = self.bert_model.embeddings.word_embeddings(input_ids)  # [num_patches, L, D]

        # 5. 将 soft prompt 作为首 token，与硬提示拼接
        soft_prompts_exp = soft_prompts.unsqueeze(1)  # [num_patches, 1, D]
        combined_embeddings = torch.cat([soft_prompts_exp, hard_prompt_embeddings], dim=1)  # [num_patches, L+1, D]

        # 6. 送入 BERT 模型处理
        outputs = self.bert_model(inputs_embeds=combined_embeddings)
        patch_embeddings = outputs.last_hidden_state.mean(dim=1)  # [num_patches, D]

        # 7. 全局汇聚所有 patch 表示，并做预测
        global_representation = patch_embeddings.mean(dim=0, keepdim=True)  # [1, D]
        prediction = self.prediction_head(global_representation)  # [1, 1]
        return prediction.squeeze()


# ---------------------------
# 消融实验变体 1：移除文本提示，仅使用软提示
# ---------------------------
class EndToEndModel_NoPrompt(nn.Module):
    def __init__(self, patch_size, emb_dim, dm, num_heads, V_prime, D, bert_model, prediction_head, stride=2):
        super(EndToEndModel_NoPrompt, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.patch_embedder = PatchEmbedder(patch_size, emb_dim, dm)
        self.prototype_reprogrammer = PrototypeReprogrammer(dm, num_heads, V_prime, D)
        self.bert_model = bert_model
        self.prediction_head = prediction_head

        # 冻结 BERT 模型参数
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def extract_patches(self, sequence):
        patches = []
        T = sequence.size(0)
        for i in range(0, T - self.patch_size + 1, self.stride):
            patches.append(sequence[i:i+self.patch_size])
        if len(patches) == 0:
            patches.append(sequence[:self.patch_size])
        return torch.stack(patches, dim=0)

    def forward(self, sequence, text_prompt=None):
        patches = self.extract_patches(sequence)
        num_patches = patches.size(0)
        patch_tokens = [self.patch_embedder(patches[i]) for i in range(num_patches)]
        patch_tokens = torch.stack(patch_tokens, dim=0)
        soft_prompts = self.prototype_reprogrammer(patch_tokens)  # [num_patches, D]
        soft_prompts = soft_prompts.unsqueeze(0)  # [1, num_patches, D]
        outputs = self.bert_model(inputs_embeds=soft_prompts)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        prediction = self.prediction_head(pooled_output)
        return prediction.squeeze()


# ---------------------------
# 消融实验变体 2：移除 reprogrammer，直接用简单线性映射
# ---------------------------
class EndToEndModel_NoReprogrammer(nn.Module):
    def __init__(self, patch_size, emb_dim, dm, D, bert_model, prediction_head, stride=2):
        super(EndToEndModel_NoReprogrammer, self).__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.patch_embedder = PatchEmbedder(patch_size, emb_dim, dm)
        self.simple_mapper = nn.Linear(dm, D)
        self.bert_model = bert_model
        self.prediction_head = prediction_head
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # 冻结 BERT 模型参数
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def extract_patches(self, sequence):
        patches = []
        T = sequence.size(0)
        for i in range(0, T - self.patch_size + 1, self.stride):
            patches.append(sequence[i:i+self.patch_size])
        if len(patches) == 0:
            patches.append(sequence[:self.patch_size])
        return torch.stack(patches, dim=0)

    def forward(self, sequence, text_prompt):
        patches = self.extract_patches(sequence)
        num_patches = patches.size(0)
        patch_tokens = [self.patch_embedder(patches[i]) for i in range(num_patches)]
        patch_tokens = torch.stack(patch_tokens, dim=0)
        mapped_tokens = self.simple_mapper(patch_tokens)  # [num_patches, D]
        inputs = self.tokenizer(text_prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].to(sequence.device)
        prompt_embeddings = self.bert_model.embeddings.word_embeddings(input_ids)
        mapped_tokens = mapped_tokens.unsqueeze(0)
        combined_embeddings = torch.cat([mapped_tokens, prompt_embeddings], dim=1)
        outputs = self.bert_model(inputs_embeds=combined_embeddings)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        prediction = self.prediction_head(pooled_output)
        return prediction.squeeze()