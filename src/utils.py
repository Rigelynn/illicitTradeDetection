# src/utils.py

import torch
import pandas as pd
import os
import dgl

def load_preprocessed_data(processed_dir):
    """加载预处理后的订单数据和违规标签"""
    ordering_path = os.path.join(processed_dir, 'ordering_processed.csv')
    fraud_path = os.path.join(processed_dir, 'fraud_labels_processed.csv')
    if not os.path.exists(ordering_path):
        raise FileNotFoundError(f"预处理后的订单数据文件未找到: {ordering_path}")
    if not os.path.exists(fraud_path):
        raise FileNotFoundError(f"预处理后的违规标签文件未找到: {fraud_path}")
    df_ordering = pd.read_csv(ordering_path)
    df_fraud = pd.read_csv(fraud_path)
    return df_ordering, df_fraud

def load_graph(processed_dir):
    """加载构建的异构图"""
    graph_path = os.path.join(processed_dir, 'hetero_graph.bin')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"异构图文件未找到: {graph_path}")
    graphs, _ = dgl.load_graphs(graph_path)
    return graphs[0]

def save_model(model, path):
    """保存模型"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """加载模型"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {path}")

def create_dataloader(X, y, batch_size=64, shuffle=True):
    """创建数据加载器"""
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader