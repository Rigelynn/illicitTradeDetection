#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline 模型代码，包括：
    - BaselineMLP：对时序数据先平均池化，再通过 MLP 进行二分类预测
    - BaselineLSTM：利用 LSTM 提取时序信息，最后用全连接层输出 logit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    def __init__(self, emb_dim, hidden_dim, output_dim=1):
        """
        emb_dim: 时序数据每个时间步的维度（例如 256）
        hidden_dim: 隐藏层维度（例如 128）
        output_dim: 输出维度，通常为 1 个 logit
        """
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        # sequence: [T, emb_dim]
        pooled = sequence.mean(dim=0)  # 平均池化得到 [emb_dim]
        x = F.relu(self.fc1(pooled))
        out = self.fc2(x)
        return out.squeeze()

class BaselineLSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers=1, output_dim=1):
        """
        emb_dim: 每个时间步的维度（例如 256）
        hidden_dim: LSTM 隐藏层维度（例如 128）
        num_layers: LSTM 层数
        output_dim: 输出维度（1）
        """
        super(BaselineLSTM, self).__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequence):
        # sequence: [T, emb_dim]；添加 batch 维度：[1, T, emb_dim]
        sequence = sequence.unsqueeze(0)
        lstm_out, _ = self.lstm(sequence)
        last_hidden = lstm_out[:, -1, :]  # 取最后时间步表示
        out = self.fc(last_hidden)
        return out.squeeze()