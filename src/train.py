#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练脚本：
    1. 加载商户 ID、违规标签、详细 prompt 以及真实时序数据
    2. 对数据做训练集 (80%) 与验证集 (20%) 划分
    3. 调用集成模型进行训练与评估，并保存训练损失曲线
"""

import os

# 设置环境变量，确保 HF_HOME 与 GPU 设备设置正确
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["HF_HOME"] = "/mnt/hdd/huggingface"

import math
import pickle
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from models.integrated_model import EndToEndModel, PredictionHead
from evaluation.evaluation import evaluate_model
from transformers import AutoTokenizer, AutoModelForCausalLM  # 使用 AutoModelForCausalLM 以保持通用性


# 定义 MerchantDataset 类
class MerchantDataset(Dataset):
    def __init__(self, merchant_ids, label_dict, prompt_csv_dir=None, sequences=None, T=25, emb_dim=256, patch_size=5,
                 stride=2):
        """
        Args:
            merchant_ids: List of merchant IDs.
            label_dict: Dictionary mapping merchant IDs to labels.
            prompt_csv_dir: Directory containing prompt CSV files.
            sequences: Dictionary mapping merchant IDs to time-series data.
            T: Time-series length.
            emb_dim: Embedding dimension per time step.
            patch_size: Number of months per patch.
            stride: Sliding window stride.
        """
        self.merchant_ids = merchant_ids
        self.labels = label_dict
        self.prompt_csv_dir = prompt_csv_dir
        self.sequences = sequences
        self.T = T
        self.emb_dim = emb_dim
        self.patch_size = patch_size
        self.stride = stride

        # Initialize prompts by reading CSV files if provided
        self.prompts = {}
        self.month_order = [
            "2021-11", "2021-12", "2022-01", "2022-02", "2022-03",
            "2022-04", "2022-05", "2022-06", "2022-07", "2022-08",
            "2022-09", "2022-10", "2022-11", "2022-12", "2023-01",
            "2023-02", "2023-03", "2023-04", "2023-05", "2023-06",
            "2023-07", "2023-08", "2023-09", "2023-10", "2023-11"
        ]
        self.default_prompt = (
            "Detailed prompt unavailable. Task: Please analyze the provided monthly order data and generate enhanced temporal features for prediction."
        )

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
        """Aggregates prompts with a sliding window approach."""
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


# 训练函数
def train_model(model, dataloader, tokenizer, optimizer, criterion, device, num_epochs=3):
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

                # Tokenize and encode aggregated_prompts
                inputs = tokenizer(aggregated_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

                # Forward pass
                pred = model(sequence, inputs)

                # Compute loss
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

    # 1. 加载商户 ID Map
    merchant_id_map_path = "/mnt/hdd/user4data/merchant_id_map.pkl"
    if os.path.exists(merchant_id_map_path):
        with open(merchant_id_map_path, "rb") as f:
            merchant_id_map = pickle.load(f)
        merchant_ids = list(merchant_id_map.keys())
    else:
        # 若没有预定义的商户 ID Map，生成示例 ID
        merchant_ids = [f"merchant_{i}" for i in range(60000)]

    # 2. 加载违规标签
    label_path = "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/merchant_overall_labels.pt"
    if os.path.exists(label_path):
        merchant_overall_labels = torch.load(label_path)
    else:
        raise ValueError("违规标签文件不存在，请核查路径：{}".format(label_path))

    # 3. 加载真实时序数据
    sequence_data_path = "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/sequences_all.pth"
    if os.path.exists(sequence_data_path):
        sequences = torch.load(sequence_data_path)
    else:
        # 若没有时序数据，使用随机数据填充
        sequences = None

    # 4. 分割数据集：80% 训练，20% 验证
    train_ids, val_ids = train_test_split(merchant_ids, test_size=0.2, random_state=42)
    print(f"Total merchants: {len(merchant_ids)} | Train: {len(train_ids)} | Val: {len(val_ids)}")

    # 5. 构造数据集与 DataLoader
    prompt_csv_dir = "/mnt/hdd/user4data/prompts"
    train_dataset = MerchantDataset(
        train_ids,
        merchant_overall_labels,
        prompt_csv_dir,
        sequences=sequences,
        T=25,
        emb_dim=256,
        patch_size=5,
        stride=2
    )
    val_dataset = MerchantDataset(
        val_ids,
        merchant_overall_labels,
        prompt_csv_dir,
        sequences=sequences,
        T=25,
        emb_dim=256,
        patch_size=5,
        stride=2
    )
    batch_size = 32
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda x: x
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=lambda x: x
    )

    # 6. 构建模型：加载 LLaMA2 模型并创建集成模型
    llama_model_path = "/mnt/hdd/huggingface/Llama-2-7b"  # 更新为正确的模型路径
    tokenizer = AutoTokenizer.from_pretrained(
        llama_model_path,
        trust_remote_code=True,
        legacy=False
    )
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    hidden_dim_pred = 512
    D = 4096  # 与 LLaMA2 隐藏层维度一致
    prediction_head = PredictionHead(D, hidden_dim_pred, output_dim=1)
    model = EndToEndModel(
        patch_size=5,
        emb_dim=256,
        dm=64,
        num_heads=4,
        V_prime=1000,
        D=D,
        llama_model=llama_model,
        prediction_head=prediction_head,
        stride=2
    )
    model.to(device)

    # 7. 定义优化器与损失函数，开始训练
    lr = 1e-4
    optimizer = torch.optim.Adam(
        list(model.patch_embedder.parameters()) +
        list(model.prototype_reprogrammer.parameters()) +
        list(model.prediction_head.parameters()),
        lr=lr
    )
    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 3  # 调试时可以设置较少的 epoch

    print("Start Training ...")
    train_losses = train_model(model, train_dataloader, tokenizer, optimizer, criterion, device, num_epochs=num_epochs)

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