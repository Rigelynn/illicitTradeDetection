#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Reprogrammer Module with Patches

该脚本利用预处理好的时序数据（文件路径为
 "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/sequences_all.pth"），
对每个商户 [25, 256] 的时序数据进行滑窗切分，生成多个 patch（例如 patch_size=5，stride=2）。
接着：
    1. 使用 PatchEmbedder 将每个 patch（形状 [patch_size, 256]）扁平化，并线性映射到 dm（例如64）维度，
       得到 patch token。
    2. 利用 PrototypeReprogrammer（见上文件）对 patch token 重编程，
       输出与预训练语言模型隐藏维度对齐的表示（D=256）。
    3. 为了提供监督信号，我们构造一个固定（不可训练）的线性映射，从 dm 到 D，
       用于生成伪目标；这相当于线性探测器，使得 reprogrammer 在已知目标空间中进行训练。

作者: Your Name
日期: 2025-02-26
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 数据及模型参数设置
SEQUENCES_FILE = "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/sequences_all.pth"
PATCH_SIZE = 5  # 每个 patch 的长度
STRIDE = 2  # 滑动步长
TIME_STEPS = 25  # 每个商户时序长度
EMBEDDING_DIM = 256  # 每个时间步的 embedding 维度

# 模型参数
dm = 64  # PatchEmbedder 输出及 PrototypeReprogrammer 输入的维度
D = 256  # PrototypeReprogrammer 输出的隐层维度（与语言模型一致）
num_heads = 4
V_prime = 1000  # 文本原型数量

# 训练超参数
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def extract_patches(sequence, patch_size, stride):
    """
    输入：
      sequence: Tensor，形状 [TIME_STEPS, EMBEDDING_DIM]
    输出：
      patches: Tensor，形状 [P, patch_size, EMBEDDING_DIM]，其中 P = floor((T - patch_size)/stride) + 1
    """
    patches = []
    T = sequence.size(0)
    for i in range(0, T - patch_size + 1, stride):
        patch = sequence[i:i + patch_size]
        patches.append(patch)
    patches = torch.stack(patches, dim=0)
    return patches


class MerchantSequenceDataset(Dataset):
    def __init__(self, sequences_file, patch_size, stride):
        """
        sequences_file: 存放所有商户时序数据的 pth 文件，格式为 dict: merchant_id -> Tensor[TIME_STEPS, EMBEDDING_DIM]
        """
        self.sequences = torch.load(sequences_file, map_location='cpu')
        self.merchant_ids = list(self.sequences.keys())
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        return len(self.merchant_ids)

    def __getitem__(self, idx):
        merchant_id = self.merchant_ids[idx]
        seq = self.sequences[merchant_id]  # [TIME_STEPS, EMBEDDING_DIM]
        patches = extract_patches(seq, self.patch_size, self.stride)  # [num_patches, patch_size, EMBEDDING_DIM]
        return patches


class PatchEmbedder(nn.Module):
    def __init__(self, patch_size, embedding_dim, dm):
        """
        将形状为 [patch_size, embedding_dim] 的 patch 扁平化后，经过线性映射到 dm 维度
        """
        super(PatchEmbedder, self).__init__()
        self.input_dim = patch_size * embedding_dim
        self.linear = nn.Linear(self.input_dim, dm)

    def forward(self, patch):
        # patch: [patch_size, embedding_dim]
        flat = patch.view(-1)  # [patch_size * embedding_dim]
        token = self.linear(flat)  # [dm]
        return token


def get_fixed_target_transform(dm, D):
    """
    构造固定（不可训练）的线性映射，从 dm 到 D，用于生成伪目标。
    """
    fixed_linear = nn.Linear(dm, D)
    for param in fixed_linear.parameters():
        param.requires_grad = False
    return fixed_linear


# 导入上面定义的 PrototypeReprogrammer 模块
from prototype_reprogrammer import PrototypeReprogrammer


def train_reprogram():
    # 初始化数据集与 DataLoader
    dataset = MerchantSequenceDataset(SEQUENCES_FILE, PATCH_SIZE, STRIDE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 初始化 PatchEmbedder 与 PrototypeReprogrammer
    patch_embedder = PatchEmbedder(PATCH_SIZE, EMBEDDING_DIM, dm).to(DEVICE)
    # 这里可以根据需要使用预训练文本 embedding 初始化原型，本文示例采用随机初始化
    proto_reprogrammer = PrototypeReprogrammer(dm=dm, num_heads=num_heads, V_prime=V_prime, D=D,
                                               init_from_pretrained=False).to(DEVICE)
    fixed_target = get_fixed_target_transform(dm, D).to(DEVICE)

    optimizer = optim.Adam(list(patch_embedder.parameters()) + list(proto_reprogrammer.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    patch_embedder.train()
    proto_reprogrammer.train()

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        total_count = 0
        for batch in dataloader:
            # batch: [B, num_patches, patch_size, EMBEDDING_DIM]
            batch = batch.to(DEVICE).float()
            B, num_patches, ps, emb_dim = batch.size()
            # 合并所有 patch, 形状变为 [B*num_patches, patch_size, EMBEDDING_DIM]
            patches = batch.view(B * num_patches, ps, emb_dim)

            # 使用 PatchEmbedder 将每个 patch 映射为 token（dm 维）
            tokens = []
            for i in range(patches.size(0)):
                token = patch_embedder(patches[i])
                tokens.append(token)
            tokens = torch.stack(tokens, dim=0)  # [B*num_patches, dm]

            # 通过固定映射生成伪目标（相当于线性探测器）：[B*num_patches, D]
            target = fixed_target(tokens)

            # Prototype Reprogramming 得到输出：[B*num_patches, D]
            output = proto_reprogrammer(tokens)

            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tokens.size(0)
            total_count += tokens.size(0)

        avg_loss = running_loss / total_count
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

    # 保存模型到指定路径
    save_dir = "/home/user4/miniconda3/projects/illicitTradeDetection/src/reprogram"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(patch_embedder.state_dict(), os.path.join(save_dir, "patch_embedder.pth"))
    torch.save(proto_reprogrammer.state_dict(), os.path.join(save_dir, "prototype_reprogrammer.pth"))
    print("训练完毕。模型已保存。")


if __name__ == "__main__":
    train_reprogram()