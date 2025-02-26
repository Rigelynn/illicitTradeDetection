#!/usr/bin/env python
# train_hgt_unsupervised.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv
from torch.cuda.amp import GradScaler, autocast

# 指定使用的 GPU（根据需要修改）
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#############################################
# 辅助函数：对商户边进行重映射
#############################################
def _remap_merchant_edges(edge_index, merchant_ids, merchant_global_map, device):
    """
    将 edge_index 中第一行（商户端）的全局商户索引，
    根据外部提供的全局映射 merchant_global_map 和当前图中保存的 merchant_ids，
    重新映射为局部编号（即在 [0, num_active_merchants) 内）。
    """
    mapping = {}
    for i, m in enumerate(merchant_ids):
        global_id = merchant_global_map.get(m, None)
        if global_id is None:
            raise ValueError(f"商户 '{m}' 不在全局映射中，请检查 merchant_id_map！")
        mapping[int(global_id)] = i

    old_indices = edge_index[0].tolist()
    new_indices = [mapping.get(idx, -1) for idx in old_indices]
    if any(i < 0 for i in new_indices):
        raise ValueError("部分商户索引没有在 merchant_ids 中找到，检查图构建过程！")
    new_edge_index = edge_index.clone()
    new_edge_index[0] = torch.tensor(new_indices, device=device)
    return new_edge_index

#############################################
# 模块1：Unsupervised HGT（仅生成商户嵌入，无后续 reprogram）
#############################################
class UnsupervisedHGT(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, embed_size, metadata,
                 num_heads=4, num_layers=2):
        """
        参数：
          in_channels_dict: 各节点类型的输入特征维度，如 {'merchant': D1, ...}
          hidden_channels: HGTConv 隐藏特征维度
          embed_size: 最终 merchant 嵌入的维度
          metadata: 图的元信息，格式为 (node_types, edge_types)
          num_heads: 注意力头数
          num_layers: HGTConv 层数
        """
        super(UnsupervisedHGT, self).__init__()
        self.num_layers = num_layers
        self.metadata = metadata

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                conv_in_channels = in_channels_dict
            else:
                conv_in_channels = {nt: hidden_channels for nt in metadata[0]}
            self.layers.append(
                HGTConv(
                    in_channels=conv_in_channels,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads
                )
            )
            self.layers.append(nn.Dropout(p=0.1))

        self.embed_layer = nn.Linear(hidden_channels, embed_size)

        # 如 "merchant" 节点原始特征维度不等于 hidden_channels，则构造投影层
        self.proj_first = nn.ModuleDict()
        if "merchant" in in_channels_dict and in_channels_dict["merchant"] != hidden_channels:
            self.proj_first["merchant"] = nn.Linear(in_channels_dict["merchant"], hidden_channels)

        # 外部提供当前图中活跃商户标识与全局映射
        self.merchant_ids = None   # 例如：['A123', 'B456', ...]
        self.merchant_global_map = None  # 例如：{'A123': 1023, 'B456': 2048, ...}

    def forward(self, x_dict, edge_index_dict):
        device = next(self.parameters()).device

        # 若设置了 merchant_ids 与全局映射，则重映射商户边
        if self.merchant_ids is not None and self.merchant_global_map is not None:
            for key in edge_index_dict.keys():
                if key[0] == "merchant":
                    edge_index_dict[key] = _remap_merchant_edges(
                        edge_index_dict[key],
                        self.merchant_ids,
                        self.merchant_global_map,
                        device
                    )

        x_dict_backup = x_dict.copy()
        for layer in self.layers:
            if isinstance(layer, HGTConv):
                out = layer(x_dict, edge_index_dict)
                if "merchant" not in out:
                    merchant_feat = x_dict["merchant"]
                    if merchant_feat.shape[1] != self.embed_layer.in_features:
                        merchant_feat = self.proj_first["merchant"](merchant_feat)
                    out["merchant"] = merchant_feat
                x_dict = {k: torch.relu(v) for k, v in out.items()}
            else:
                x_dict = {k: layer(v) for k, v in x_dict.items()}

        merchant_features = x_dict.get("merchant", x_dict_backup.get("merchant"))
        if merchant_features.shape[1] != self.embed_layer.in_features:
            raise KeyError(
                f"'merchant' 节点特征维度为 {merchant_features.shape[1]}，期望为 {self.embed_layer.in_features}")
        embeddings = self.embed_layer(merchant_features)
        return embeddings

#############################################
# 训练、数据加载及主流程
#############################################
def train_unsupervised(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        with autocast():
            # 获取 HGT 生成的无监督嵌入
            embeddings = model(batch.x_dict, batch.edge_index_dict)
            # 示例无监督损失：最小化嵌入向量的 L2 范数（实际任务中可更换为合适的损失函数）
            loss = torch.mean(embeddings ** 2)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def prepare_dataset_by_month(graph_dir):
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
        except Exception:
            continue
        monthly_datasets.setdefault(month_str, []).append(graph)
    return monthly_datasets

def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # 图数据存放目录（按月份组织的 PyG 异构图）
    graph_dir = "/mnt/hdd/user4data/pyg_graphs"
    logging.info("加载按月份组织的 PyG 图数据...")
    monthly_datasets = prepare_dataset_by_month(graph_dir)
    if not monthly_datasets:
        logging.error("未找到任何图数据，请检查目录或文件格式。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50

    # 模型参数设置
    hidden_channels = 128
    embed_size = 256
    num_heads = 8
    num_layers = 4

    # 加载全局商户映射（merchant_id_map）路径更新至 /mnt/hdd/user4data/ 目录下
    merchant_map_path = "/mnt/hdd/user4data/merchant_id_map.pkl"
    if os.path.exists(merchant_map_path):
        try:
            with open(merchant_map_path, 'rb') as f:
                merchant_global_map = pickle.load(f)
            logging.info("成功加载 merchant_id_map.pkl")
        except Exception as e:
            logging.error(f"加载 merchant_id_map.pkl 时出错：{e}")
            return
    else:
        logging.error(f"merchant_map_path 未找到: {merchant_map_path}")
        return

    # 按月份处理数据
    for month, graphs in sorted(monthly_datasets.items()):
        logging.info(f"----- 开始处理月份 {month} -----")
        if not graphs:
            logging.warning(f"月份 {month} 没有有效图数据，跳过。")
            continue

        loader = DataLoader(
            graphs,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        sample_graph = graphs[0]
        metadata = sample_graph.metadata()  # (node_types, edge_types)
        in_channels_dict = {nt: sample_graph[nt].x.size(1) for nt in sample_graph.node_types}

        # 初始化 HGT 模型（无监督嵌入模块）
        hgt_model = UnsupervisedHGT(
            in_channels_dict,
            hidden_channels,
            embed_size,
            metadata,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)
        # 设置模型所需的 merchant_ids 与全局映射
        if hasattr(sample_graph, 'merchant_ids'):
            hgt_model.merchant_ids = sample_graph.merchant_ids
            hgt_model.merchant_global_map = merchant_global_map
        else:
            logging.warning("当前图数据中没有 merchant_ids 属性！")

        optimizer = optim.Adam(hgt_model.parameters(), lr=0.001)
        scaler = GradScaler()

        for epoch in range(1, epochs + 1):
            logging.info(f"{month} - Epoch {epoch}/{epochs} 开始训练...")
            train_loss = train_unsupervised(hgt_model, loader, optimizer, device, scaler)
            logging.info(f"{month} - Epoch {epoch}: Train Loss = {train_loss:.4f}")
            torch.cuda.empty_cache()

        # 使用 HGT 模型生成无监督嵌入，并保存结果
        hgt_model.eval()
        monthly_embeddings = {}
        with torch.no_grad():
            for data in graphs:
                data = data.to(device)
                embeddings = hgt_model(data.x_dict, data.edge_index_dict)
                if hasattr(data, 'merchant_ids'):
                    if len(data.merchant_ids) != embeddings.size(0):
                        logging.warning("merchant_ids 数量与嵌入数量不匹配！")
                    for mid, emb in zip(data.merchant_ids, embeddings):
                        monthly_embeddings[str(mid)] = emb.cpu()
                else:
                    for idx, emb in enumerate(embeddings):
                        monthly_embeddings[f"merchant_{idx}"] = emb.cpu()

        # 将检查点存放在 /mnt/hdd/user4data/checkpoints 下
        checkpoint_dir = os.path.join("/mnt/hdd/user4data/checkpoints", month)
        os.makedirs(checkpoint_dir, exist_ok=True)
        embedding_file = os.path.join(checkpoint_dir, f"monthly_embeddings_{month}.pth")
        try:
            torch.save(monthly_embeddings, embedding_file)
            logging.info(f"{month} - 无监督嵌入已保存到: {embedding_file}")
        except Exception as e:
            logging.error(f"保存嵌入失败：{e}")

if __name__ == "__main__":
    main()