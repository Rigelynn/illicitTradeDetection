# src/utils.py

import torch
import pandas as pd
import os
import dgl
import logging
import sys
import numpy as np
from logging.handlers import RotatingFileHandler


def setup_logging(log_file='logs/data_processing.log', max_bytes=10**6, backup_count=5, log_level=logging.INFO):
    """
    设置日志记录格式和级别，并启用日志轮转。

    参数：
    - log_file: str，日志文件的路径。
    - max_bytes: int，日志文件的最大字节数，超过则轮转。
    - backup_count: int，保留的旧日志文件的数量。
    - log_level: int，日志级别，如 logging.DEBUG, logging.INFO 等。
    """
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 获取根记录器
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 清空之前添加的处理器，避免重复日志显示
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 创建控制台日志处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建文件日志处理器，并启用日志轮转
    file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

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
    logging.info(f"Model saved to {path}")


def load_model(model, path, device):
    """加载模型"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    logging.info(f"Model loaded from {path}")


def create_dataloader(X, y, batch_size=64, shuffle=True):
    """创建数据加载器"""
    if isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values, dtype=torch.float32)
    elif isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)

    if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = torch.tensor(y.values, dtype=torch.long).squeeze()
    elif isinstance(y, np.ndarray):
        y = torch.tensor(y, dtype=torch.long).squeeze()

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def save_dataframe(df, filename, processed_dir):
    """保存 DataFrame 为 CSV 文件"""
    try:
        os.makedirs(processed_dir, exist_ok=True)
        path = os.path.join(processed_dir, filename)
        df.to_csv(path, index=False)
        logging.info(f"数据已保存到 {path}")
    except Exception as e:
        logging.error(f"保存数据时发生错误：{e}")
        raise