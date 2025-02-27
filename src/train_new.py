#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本说明：
1. 加载数据与标签，并进行数据集划分
2. 构造数据加载器
3. 加载 BERT 模型（替换原 Llama2 模型）并构建集成模型
4. 训练、评估并绘制训练损失曲线
"""

import os
import math
import pickle
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from models.model import EndToEndModel, PredictionHead
from evaluation.evaluation import evaluate_model
from transformers import BertTokenizer, BertModel


# 定义 MerchantDataset 类
class MerchantDataset(Dataset):
    def __init__(self, merchant_ids, label_dict, prompt_csv_dir=None, sequences=None, T=25, emb_dim=256, patch_size=5,
                 stride=2):
        """
        Args:
            merchant_ids: 商户ID列表。
            label_dict: 商户ID到标签的映射字典。
            prompt_csv_dir: 保存提示信息的CSV文件夹路径。
            sequences: 商户ID到时序数据（tensor）的映射。
            T: 时序长度。
            emb_dim: 每个时间步的特征维度。
            patch_size: 分块窗口大小。
            stride: 滑动步长。
        """
        self.merchant_ids = merchant_ids
        self.labels = label_dict
        self.prompt_csv_dir = prompt_csv_dir
        self.sequences = sequences
        self.T = T
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.stride = stride

        self.prompts = {}
        self.month_order = [
            "2021-11", "2021-12", "2022-01", "2022-02", "2022-03",
            "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
            "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
            "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
            "2023-07", "2023-08", "2023-09", "2023-10", "2023-11"
        ]
        self.default_prompt = ("Detailed prompt unavailable. Task: Please analyze the provided "
                               "monthly order data and generate enhanced temporal features for prediction.")

        if prompt_csv_dir and os.path.exists(prompt_csv_dir):
            for file in os.listdir(prompt_csv_dir):
                if file.startswith("detailed_prompts_") and file.endswith(".csv"):
                    df_prompts = pd.read_csv(os.path.join(prompt_csv_dir, file))
                    for idx, row in df_prompts.iterrows():
                        merchant_id = row['merchant_id']
                        month = row.get('month', None)
                        prompt_text = row['prompt']
                        if merchant_id not in self.prompts:
                            self.prompts[merchant_id] = {}
                        if month:
                            self.prompts[merchant_id][month] = prompt_text

    def aggregate_prompts(self, monthly_prompts):
        """利用滑动窗口聚合月度提示信息"""
        aggregated = []
        L = len(monthly_prompts)
        for i in range(0, L - self.patch_size + 1, self.stride):
            window_prompts = monthly_prompts[i:i + self.patch_size]
            agg_text = " ; ".join(window_prompts)
            aggregated.append(agg_text)
        if len(aggregated) == 0:
            agg_text = " ; ".join(monthly_prompts[:self.patch_size])
            aggregated.append(agg_text)
        return aggregated

    def __len__(self):
        return len(self.merchant_ids)

    def __getitem__(self, idx):
        merchant_id = self.merchant_ids[idx]
        if self.sequences and merchant_id in self.sequences:
            sequence = self.sequences[merchant_id].to(torch.float32)
        else:
            sequence = torch.randn(self.T, self.emb_dim)

        monthly_prompts = []
        for month in self.month_order:
            if merchant_id in self.prompts and month in self.prompts[merchant_id]:
                monthly_prompts.append(self.prompts[merchant_id][month])
            else:
                monthly_prompts.append(self.default_prompt)

        aggregated_prompts = self.aggregate_prompts(monthly_prompts)
        label = self.labels.get(merchant_id, 0)

        return merchant_id, sequence, aggregated_prompts, torch.tensor(label, dtype=torch.float32)


def train_model(model, dataloader, optimizer, criterion, device, num_epochs=3):
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            loss_batch = 0.0
            for sample in batch:
                merchant_id, sequence, aggregated_prompts, label = sample
                sequence = sequence.to(device)
                label = label.to(device)

                # 根据标签动态设置 custom_stride：
                # fraud（正样本，label==1）：细粒度（custom_stride=1），正常样本：粗粒度（custom_stride=patch_size）
                if label.item() == 1:
                    custom_stride = 1
                else:
                    custom_stride = model.patch_size  # 这里使用 patch_size 作为较大步长

                pred = model(sequence, aggregated_prompts, custom_stride=custom_stride)
                loss = criterion(pred.squeeze(), label)
                loss.backward()
                loss_batch += loss.item()
            optimizer.step()
            running_loss += loss_batch / len(batch)
        avg_loss = running_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {avg_loss:.6f}")
    return train_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. 加载商户ID映射
    merchant_id_map_path = "/mnt/hdd/user4data/merchant_id_map.pkl"
    if os.path.exists(merchant_id_map_path):
        with open(merchant_id_map_path, "rb") as f:
            merchant_id_map = pickle.load(f)
        merchant_ids = list(merchant_id_map.keys())
    else:
        merchant_ids = [f"merchant_{i}" for i in range(60000)]

    # 2. 加载标签
    label_path = "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/merchant_overall_labels.pt"
    if os.path.exists(label_path):
        merchant_overall_labels = torch.load(label_path)
    else:
        raise ValueError("违规标签文件不存在，请核查路径：{}".format(label_path))

    # 3. 加载时序数据
    sequence_data_path = "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/sequences_all.pth"
    if os.path.exists(sequence_data_path):
        sequences = torch.load(sequence_data_path)
    else:
        sequences = None

    # 4. 划分数据集
    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(merchant_ids, test_size=0.2, random_state=42)
    print(f"Total merchants: {len(merchant_ids)} | Train: {len(train_ids)} | Val: {len(val_ids)}")

    # 5. 构造数据集与 DataLoader
    prompt_csv_dir = "/mnt/hdd/user4data/prompts"
    train_dataset = MerchantDataset(train_ids, merchant_overall_labels, prompt_csv_dir, sequences=sequences, T=25,
                                    emb_dim=256, patch_size=5, stride=2)
    val_dataset = MerchantDataset(val_ids, merchant_overall_labels, prompt_csv_dir, sequences=sequences, T=25,
                                  emb_dim=256, patch_size=5, stride=2)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                  collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True,
                                collate_fn=lambda x: x)

    # 6. 加载 BERT 模型及分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # 构建集成模型，并设置预测头的参数
    hidden_dim_pred = 512
    D = 768  # 对于 bert-base-uncased，隐藏层维度为 768
    prediction_head = PredictionHead(D, hidden_dim_pred, output_dim=1)
    model = EndToEndModel(
        patch_size=5,
        emb_dim=256,
        dm=64,
        num_heads=4,
        V_prime=1000,
        D=D,
        bert_model=bert_model,
        prediction_head=prediction_head,
        stride=2
    )
    model.to(device)

    # 7. 定义优化器与损失函数
    lr = 1e-4
    optimizer = torch.optim.Adam(
        list(model.patch_embedder.parameters()) +
        list(model.prototype_reprogrammer.parameters()) +
        list(model.prediction_head.parameters()),
        lr=lr
    )
    # 设置正样本权重（例如 3.0），缓解类别不平衡
    pos_weight = torch.tensor(3.0).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    num_epochs = 3
    print("Start Training ...")
    train_losses = train_model(model, train_dataloader, optimizer, criterion, device, num_epochs=num_epochs)

    # 8. 模型评估
    accuracy, precision, recall, f1, auc, cm = evaluate_model(model, val_dataloader, device)
    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # 9. 绘制并保存训练损失曲线
    epochs_range = list(range(1, num_epochs + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_losses, marker='o', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.show()


if __name__ == "__main__":
    main()