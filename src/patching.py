#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务2：Patching
采用 TimeLLM 的方法，将归一化后的时序划分为多个 patch。
"""

import torch

def create_patches(sequence, patch_size, stride):
    """
    将时序 tensor 按照指定 patch_size 和 stride 划分为多个 patch。

    Args:
        sequence (torch.Tensor): 形状 [T, embed_dim]
        patch_size (int): 每个 patch 包含的时间步数
        stride (int): 滑动步幅
    Returns:
        List[torch.Tensor]: 每个 patch 的 tensor, 形状 [patch_size, embed_dim]
    """
    seq_len = sequence.shape[0]
    patches = []
    for i in range(0, seq_len - patch_size + 1, stride):
        patch = sequence[i: i + patch_size, :]
        patches.append(patch)
    return patches

def main():
    # 加载已归一化的时序数据
    sequence = torch.load("sequence.pt")
    patch_size = 5            # 例如，每个 patch 包含 5 个时间步
    stride = patch_size // 2  # 使用重叠 patch（stride = patch_size/2）
    patches = create_patches(sequence, patch_size=patch_size, stride=stride)
    print("生成的 patches 数量:", len(patches))
    # 保存 patch 数据，后续任务中加载使用
    torch.save(patches, "patches.pt")
    print("Patches 已保存到 'patches.pt'.")

if __name__ == "__main__":
    main()