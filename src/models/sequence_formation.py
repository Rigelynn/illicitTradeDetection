#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sequence_formation.py

用途：
    从 /mnt/hdd/user4data/checkpoints 下加载 2021-11 到 2023-11 这 25 个月的 HGT 商户 embedding，
    数据格式均为字典：键为商户 ID（例如 '410101100009'），值为 embedding 向量（形状为 (256,)）。

    脚本会遍历所有月份，构造每个商户的时序数据：
      每个商户对应一个长度为 25 的列表，若某个月份该商户缺失，则用零向量补充。

    采用 RevIN 风格归一化：
      对每个商户形成的时序（形状 [25, 256]）进行归一化，归一化过程是先将所有商户的
      时序堆叠成 [B, 25, 256]，然后对每个样本在时间维度上计算均值和标准差完成归一化。

    最终结果将保存为单个文件，保存在 /home/user4/miniconda3/projects/illicitTradeDetection/data/processed 下，
    文件名为 sequences_all.pth。

预估大小：
    每个商户的时序数据大约 25KB，总体数据大约 60000 × 25KB ≈ 1.5GB（加上字典存储开销）。
"""
import os
import torch

# 配置参数
BASE_PATH = "/mnt/hdd/user4data/checkpoints"
# 输出目录更新为 /home/user4/miniconda3/projects/illicitTradeDetection/data/processed
OUTPUT_DIR = "/data/processed"
OUTPUT_FILE = "sequences_all.pth"
BATCH_SIZE = 10000  # 用于归一化的批处理大小，可以根据内存调整


def get_month_list(base_path):
    """
    返回 base_path 下的月份文件夹名称（如 '2021-11', '2021-12', ..., '2023-11'），
    排序按字典序（应与时间顺序一致）。
    """
    month_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    month_dirs = sorted(month_dirs)
    return month_dirs


def load_month_embeddings(month):
    """
    加载指定月份的 embedding 文件。

    参数:
        month (str): 如 '2021-11'
    返回:
        data: 字典类型，键为商户ID，值为对应 tensor（形状 (256,)）
    """
    file_path = os.path.join(BASE_PATH, month, f"monthly_embeddings_{month}.pth")
    if not os.path.exists(file_path):
        print(f"[WARNING] 文件不存在: {file_path}")
        return None
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"加载 {file_path} 时出错: {e}")
        return None
    return data


def build_merchant_sequences(month_list):
    """
    遍历每个月份，构造一个字典，其中 key 为商户 ID，
    value 为长度等于月份数（本例中为25）的列表，每个元素为对应月份的 embedding。
    若某个月份中该商户缺失，则用零向量补充。

    同时根据第一次遇到的有效商户 embedding 确定 embed_dim。

    返回:
        sequences: dict，商户 ID -> list of 25 个 embedding（稍后堆叠为 tensor）
        embed_dim: embedding 的维度（如256）
    """
    num_months = len(month_list)
    sequences = {}  # key: merchant_id, value: list of length num_months（初始均为 None）
    embed_dim = None

    for month_idx, month in enumerate(month_list):
        print(f"正在处理月份: {month} [{month_idx + 1}/{num_months}]")
        data = load_month_embeddings(month)
        if data is None:
            print(f"跳过月份 {month}（未加载到数据）")
            continue

        # 本月数据均为字典格式处理
        for merchant_id, emb in data.items():
            if embed_dim is None and isinstance(emb, torch.Tensor):
                embed_dim = emb.size(0)
                print(f"确定 embedding 维度为: {embed_dim}")
            if merchant_id not in sequences:
                sequences[merchant_id] = [None] * num_months
            sequences[merchant_id][month_idx] = emb

    if embed_dim is None:
        raise ValueError("未能从数据中确定 embedding 维度，请检查数据文件。")

    # 后处理：填充缺失月份（用零向量）并堆叠为 tensor [num_months, embed_dim]
    for merchant_id, emb_list in sequences.items():
        for idx in range(len(emb_list)):
            if emb_list[idx] is None:
                emb_list[idx] = torch.zeros(embed_dim)
        sequences[merchant_id] = torch.stack(emb_list, dim=0)

    return sequences, embed_dim


def revin_normalize_all(sequences, batch_size):
    """
    对所有商户的时序数据进行 RevIN 风格的归一化。

    参数:
        sequences: dict，商户 ID -> tensor [25, 256]
        batch_size: int，用于批处理的大小
    返回:
        normalized_sequences: dict，商户 ID -> 归一化后的 tensor [25, 256]
    """
    merchant_ids = list(sequences.keys())
    total = len(merchant_ids)
    normalized_sequences = {}

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_ids = merchant_ids[start:end]
        batch_list = [sequences[mid] for mid in batch_ids]
        batch_tensor = torch.stack(batch_list, dim=0)  # [B, 25, 256]

        # [B, 25, 256] -> [B, 256, 25]
        batch_trans = batch_tensor.transpose(1, 2)
        mean = batch_trans.mean(dim=2, keepdim=True)
        std = batch_trans.std(dim=2, keepdim=True) + 1e-5
        batch_norm = (batch_trans - mean) / std
        # [B, 256, 25] -> [B, 25, 256]
        batch_norm = batch_norm.transpose(1, 2)

        # 存储归一化后的数据
        for i, mid in enumerate(batch_ids):
            normalized_sequences[mid] = batch_norm[i]

        print(f"已归一化并处理批次 {start // batch_size + 1} / {(total + batch_size - 1) // batch_size}")

    return normalized_sequences


def save_all_sequences(sequences, output_dir, output_file):
    """
    将所有商户的时序数据保存为单个文件。

    参数:
        sequences: dict，商户 ID -> 归一化后的 tensor [25, 256]
        output_dir: str，保存目录
        output_file: str，保存文件名
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, output_file)
    torch.save(sequences, output_path)
    print(f"所有商户的时序数据已保存到 {output_path}")


def main():
    month_list = get_month_list(BASE_PATH)
    if not month_list:
        print(f"在路径 {BASE_PATH} 下未找到月份文件夹，请检查路径设置。")
        return
    print("检测到的月份：", month_list)

    sequences, embed_dim = build_merchant_sequences(month_list)
    num_months = len(month_list)
    print(f"共形成 {len(sequences)} 个商户的时序，每个时序形状为 [{num_months}, {embed_dim}]")

    print("开始进行归一化处理...")
    normalized_sequences = revin_normalize_all(sequences, BATCH_SIZE)

    print("开始保存归一化后的时序数据...")
    save_all_sequences(normalized_sequences, OUTPUT_DIR, OUTPUT_FILE)
    print("所有商户时序归一化并保存完毕。")


if __name__ == "__main__":
    main()