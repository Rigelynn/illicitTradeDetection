#!/usr/bin/env python
# timeseries_encoders.py

import os
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------
# 辅助函数：加载并堆叠各月的 prompt tokens
# ---------------------------
def load_prompt_tokens(checkpoints_dir):
    """
    假设 checkpoints_dir 下每个文件夹的名称为月份（例如 "2021-11"），
    每个文件夹中都有形如 monthly_prompt_tokens_{month}.pth 的文件，
    文件内容为形状 [num_merchants, embed_dim] 的 tensor，
    各月商户的顺序一致。
    返回的 prompt_series 形状为 [num_merchants, num_months, embed_dim]
    """
    months = sorted(os.listdir(checkpoints_dir))
    prompt_tokens = []
    for month in months:
        token_file = f"monthly_prompt_tokens_{month}.pth"
        file_path = os.path.join(checkpoints_dir, month, token_file)
        if os.path.exists(file_path):
            try:
                tokens = torch.load(file_path)
                prompt_tokens.append(tokens)
            except Exception as e:
                print(f"加载 {file_path} 失败: {e}")
        else:
            print(f"未找到 {file_path}")
    if not prompt_tokens:
        raise ValueError("未加载到任何 prompt tokens！")
    # 将月份堆叠到时间轴上：最终形状为 [num_merchants, num_months, embed_dim]
    prompt_series = torch.stack(prompt_tokens, dim=1)
    return prompt_series


# ---------------------------
# 定义时序编码器模型
# ---------------------------
# 1. LSTM-based Encoder
class LSTMTimeSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMTimeSeries, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]，其中 batch 为商户个数
        out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 2. GRU-based Encoder
class GRUTimeSeries(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUTimeSeries, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# 3. Transformer-based Encoder
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(TransformerTimeSeries, self).__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.input_fc(x)  # [batch, seq_len, model_dim]
        # Transformer 的输入格式为 [seq_len, batch, model_dim]
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        # 取最后一个时间步的输出
        x = x[-1, :, :]
        out = self.fc(x)
        return out


# ---------------------------
# 时间序列模型训练函数
# ---------------------------
def train_time_series_model(model, series, labels, epochs=50, lr=0.001, device=torch.device("cpu")):
    """
    model: 时序编码器（如 LSTMTimeSeries 等）
    series: [num_merchants, num_months, embed_dim]
    labels: tensor [num_merchants]，二分类标签 (0/1)
    """
    model.to(device)
    series = series.to(device)
    labels = labels.to(device).float()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(series)  # 预期输出 [num_merchants, 1]
        outputs = outputs.squeeze()
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    checkpoints_dir = "/home/user4/miniconda3/projects/illicitTradeDetection/src/checkpoints"
    series = load_prompt_tokens(checkpoints_dir)

    # 假设每个商户的标签从某处加载；这里为了示例，我们随机生成二分类标签
    num_merchants = series.size(0)
    labels = torch.randint(0, 2, (num_merchants,))

    input_dim = series.size(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型1：LSTM
    lstm_model = LSTMTimeSeries(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1)
    print("Training LSTM Time Series Model:")
    train_time_series_model(lstm_model, series, labels, epochs=50, lr=0.001, device=device)

    # 模型2：GRU
    gru_model = GRUTimeSeries(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1)
    print("Training GRU Time Series Model:")
    train_time_series_model(gru_model, series, labels, epochs=50, lr=0.001, device=device)

    # 模型3：Transformer
    transformer_model = TransformerTimeSeries(input_dim=input_dim, model_dim=64, num_heads=4, num_layers=2,
                                              output_dim=1, dropout=0.1)
    print("Training Transformer Time Series Model:")
    train_time_series_model(transformer_model, series, labels, epochs=50, lr=0.001, device=device)