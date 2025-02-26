#!/usr/bin/env python
# experiment1_dynamic_neural_networks.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import pickle

# 导入TGAT和EvolveGCN模型，假设你已经有这两个模型的实现
from tgaten import TGAT  # 请根据实际路径导入
from evolvegcn import EvolveGCN  # 请根据实际路径导入

# 设置可见GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#############################################
# 实验1: 动态神经网络方法（TGAT & EvolveGCN）
#############################################

def prepare_dataset_by_month(graph_dir):
    """
    加载按月份组织的PyG异构图数据
    """
    monthly_datasets = {}
    for fname in os.listdir(graph_dir):
        if not fname.endswith(".pt"):
            continue
        try:
            month_str = fname.split("_")[-1].split(".")[0]
        except Exception:
            continue
        graph_path = os.path.join(graph_dir, fname)
        try:
            graph = torch.load(graph_path)
        except Exception as e:
            print(f"加载图文件 {graph_path} 失败: {e}")
            continue
        monthly_datasets.setdefault(month_str, []).append(graph)
    return monthly_datasets

def train_model(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # 假设有标签存在于batch.y
        loss = loss_fn(out, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            preds.extend(torch.sigmoid(out).cpu().numpy())
            labels.extend(batch.y.cpu().numpy())
    preds_binary = [1 if p >= 0.5 else 0 for p in preds]
    acc = accuracy_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary, zero_division=0)
    try:
        auc = roc_auc_score(labels, preds)
    except Exception:
        auc = 0.0
    precision = precision_score(labels, preds_binary, zero_division=0)
    recall = recall_score(labels, preds_binary, zero_division=0)
    return acc, f1, auc, precision, recall

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    graph_dir = "/mnt/hdd/user4data/pyg_graphs"
    logging.info("加载按月份组织的 PyG 异构图...")
    monthly_datasets = prepare_dataset_by_month(graph_dir)
    if not monthly_datasets:
        logging.error("未找到任何图数据，请检查目录或文件格式。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50

    # 假设所有图的元信息相同，以第一个图为例
    sample_month = sorted(monthly_datasets.keys())[0]
    sample_graph = monthly_datasets[sample_month][0]
    in_channels = sample_graph.x.size(1)
    num_classes = 1  # 二分类

    # 实验1.1: TGAT模型
    logging.info("初始化TGAT模型...")
    tgaten_model = TGAT(in_channels=in_channels, hidden_channels=128, num_classes=num_classes).to(device)
    tgaten_optimizer = optim.Adam(tgaten_model.parameters(), lr=0.001)
    tgaten_loss_fn = nn.BCEWithLogitsLoss()

    # 实验1.2: EvolveGCN模型
    logging.info("初始化EvolveGCN模型...")
    evolvegcn_model = EvolveGCN(in_channels=in_channels, hidden_channels=128, num_classes=num_classes, num_layers=3).to(device)
    evolvegcn_optimizer = optim.Adam(evolvegcn_model.parameters(), lr=0.001)
    evolvegcn_loss_fn = nn.BCEWithLogitsLoss()

    # 准备数据加载器
    for month, graphs in sorted(monthly_datasets.items()):
        logging.info(f"----- 开始处理月份 {month} -----")
        if not graphs:
            logging.warning(f"月份 {month} 没有有效图数据，跳过。")
            continue

        loader = DataLoader(graphs, batch_size=32, shuffle=True)

        # 训练TGAT
        logging.info(f"训练TGAT模型，月份 {month}...")
        for epoch in range(1, epochs + 1):
            loss = train_model(tgaten_model, loader, tgaten_optimizer, device, tgaten_loss_fn)
            if epoch % 10 == 0:
                logging.info(f"TGAT - Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

        # 训练EvolveGCN
        logging.info(f"训练EvolveGCN模型，月份 {month}...")
        for epoch in range(1, epochs + 1):
            loss = train_model(evolvegcn_model, loader, evolvegcn_optimizer, device, evolvegcn_loss_fn)
            if epoch % 10 == 0:
                logging.info(f"EvolveGCN - Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    # 评估模型（假设有验证集）
    # 这里需要加载验证集的图数据，并进行评估
    # 例如：
    # val_loader = DataLoader(validation_graphs, batch_size=32, shuffle=False)
    # acc, f1, auc, precision, recall = evaluate_model(tgaten_model, val_loader, device)
    # logging.info(f"TGAT Validation - Acc: {acc}, F1: {f1}, AUC: {auc}, Precision: {precision}, Recall: {recall}")
    # 同理评估EvolveGCN模型

    # 可视化训练过程中的损失和指标（假设有记录）
    # 这里需根据实际情况记录每个epoch的loss和指标，再进行绘图

if __name__ == "__main__":
    main()