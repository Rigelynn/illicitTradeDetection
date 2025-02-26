#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_embeddings.py

用途：
    检查存储在 /mnt/hdd/user4data/checkpoints 下各个月份的商户 embedding 信息。
    每个文件夹对应一个月份（例如 '2021-11'），文件名为 monthly_embeddings_<month>.pth。
    脚本会打印出每个月的 embedding 数量、数据类型以及部分样本信息，方便了解数据情况。
"""

import os
import torch

def check_embeddings(base_path):
    # 获取所有文件夹名称（假设文件夹名称格式为 YYYY-MM）
    month_dirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if not month_dirs:
        print("未在 {} 下找到任何月份文件夹。".format(base_path))
        return

    for month in month_dirs:
        # 构造文件路径，例如 /mnt/hdd/user4data/checkpoints/2021-11/monthly_embeddings_2021-11.pth
        file_path = os.path.join(base_path, month, f"monthly_embeddings_{month}.pth")
        if not os.path.exists(file_path):
            print(f"[WARNING] 文件不存在: {file_path}")
            continue

        print("=" * 60)
        print(f"正在处理 {file_path}")

        try:
            data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
            continue

        if isinstance(data, torch.Tensor):
            # 假设 tensor 形状为 [num_merchants, embed_dim]
            num_merchants = data.size(0)
            embed_dim = data.size(1) if data.dim() > 1 else None
            print(f"月份 {month}: Tensor 格式，商户数量 = {num_merchants}，embedding 维度 = {embed_dim}")
        elif isinstance(data, dict):
            num_merchants = len(data)
            sample_key = list(data.keys())[0] if num_merchants > 0 else None
            sample_shape = None
            if sample_key is not None and isinstance(data[sample_key], torch.Tensor):
                sample_shape = tuple(data[sample_key].shape)
            print(f"月份 {month}: Dictionary 格式，商户数量 = {num_merchants}，示例 key = {sample_key}，示例 embedding 形状 = {sample_shape}")
        else:
            print(f"月份 {month}: 未识别的数据格式：{type(data)}")

    print("=" * 60)

if __name__ == "__main__":
    base_path = "/mnt/hdd/user4data/checkpoints"
    check_embeddings(base_path)