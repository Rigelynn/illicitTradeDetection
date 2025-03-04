#!/usr/bin/env python
# train_supervised_hgt.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import pickle
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

#############################################
# 辅助函数：对商户边进行重映射
#############################################

def _remap_merchant_edges(edge_index, merchant_ids, merchant_global_map, device):
    """
    将 edge_index 中第一行（商户端）的全局商户索引，
    根据 merchant_global_map 和当前图中的 merchant_ids，
    重新映射为局部编号。
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
# 有监督 HGT 模型定义（不生成 prompt tokens）
#############################################

class SupervisedHGT(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, embed_size, metadata,
                 num_heads=4, num_layers=2):
        """
        参数：
          in_channels_dict: 各节点类型的输入特征维度，如 {'merchant': num_features, 'goods': num_features}
          hidden_channels: HGTConv 隐藏特征维度
          embed_size: 投影后（同时作为分类输入）的维度
          metadata: 图的元信息，格式为 (node_types, edge_types)
          num_heads: 注意力头数
          num_layers: HGTConv 层数
        """
        super(SupervisedHGT, self).__init__()
        self.num_layers = num_layers
        self.metadata = metadata

        # 构建多层 HGTConv + Dropout
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

        # 投影至 embed_size
        self.embed_layer = nn.Linear(hidden_channels, embed_size)

        # 分类器，用于对 merchant 进行标签预测（二分类：0/1）
        self.classifier = nn.Linear(embed_size, 1)

        # 如果 'merchant' 特征维度不等于 hidden_channels，则构造投影层
        self.proj_first = nn.ModuleDict()
        if "merchant" in in_channels_dict and in_channels_dict["merchant"] != hidden_channels:
            self.proj_first["merchant"] = nn.Linear(in_channels_dict["merchant"], hidden_channels)

        # 外部传入：当前图中活跃商户的标识列表和全局映射
        self.merchant_ids = None       # 例如：['A123', 'B456', ...]
        self.merchant_global_map = None  # 例如：{'A123': 1023, 'B456': 2048, ...}

    def forward(self, x_dict, edge_index_dict):
        device = next(self.parameters()).device

        # 对 "merchant" 节点对应的边进行重新映射
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
                f"'merchant' 节点特征维度为 {merchant_features.shape[1]}，期望为 {self.embed_layer.in_features}"
            )
        embeddings = self.embed_layer(merchant_features)
        # 分类分支：对 merchant embedding 进行二分类预测
        logits = self.classifier(embeddings)  # 输出 shape [N, 1]
        return embeddings, logits

#############################################
# 训练、数据加载及主流程
#############################################

def train_supervised(model, loader, optimizer, device, scaler, merchant_labels, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        # 优先使用在 main() 中赋值给 model 的 merchant_ids，
        # 以避免 DataLoader 批处理后丢失原有的所有 merchant id 信息
        if model.merchant_ids is not None:
            merchant_id_list = model.merchant_ids
        elif hasattr(batch, 'merchant_ids'):
            merchant_id_list = batch.merchant_ids
        else:
            logging.error("当前 batch 缺少 merchant_ids 属性！")
            continue

        # 根据 merchant_id_list 生成标签，确保标签数量与 logits 数量一致
        y = torch.tensor(
            [float(merchant_labels.get(str(mid), 0)) for mid in merchant_id_list],
            dtype=torch.float32,
            device=device
        )

        optimizer.zero_grad()
        with autocast():
            embeddings, logits = model(batch.x_dict, batch.edge_index_dict)
            logits = logits.view(-1)
            # 检查 logits 数量是否与标签数量一致，便于调试
            if logits.shape[0] != y.shape[0]:
                raise ValueError(f"Logits 数量与标签数量不匹配: logits: {logits.shape[0]}, labels: {y.shape[0]}")
            loss = criterion(logits, y)
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

    # 图数据文件和 merchant_id_map 均放新的路径下
    graph_dir = "/mnt/hdd/user4data/pyg_graphs"
    logging.info("加载按月份组织的 PyG 异构图...")
    monthly_datasets = prepare_dataset_by_month(graph_dir)
    if not monthly_datasets:
        logging.error("未找到任何图数据，请检查目录或文件格式。")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50

    # 模型参数
    hidden_channels = 128
    embed_size = 256
    num_heads = 8
    num_layers = 4

    # 加载全局商户映射，使用新的路径
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

    # 加载有标签文件，标签仍然在原来的目录下
    merchant_label_path = "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/merchant_overall_labels.pt"
    if os.path.exists(merchant_label_path):
        try:
            merchant_labels = torch.load(merchant_label_path)
            logging.info("成功加载 merchant_overall_labels.pt")
        except Exception as e:
            logging.error(f"加载 merchant_overall_labels.pt 时出错：{e}")
            return
    else:
        logging.error(f"merchant_label_path 未找到: {merchant_label_path}")
        return

    criterion = nn.BCEWithLogitsLoss()

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

        model = SupervisedHGT(
            in_channels_dict,
            hidden_channels,
            embed_size,
            metadata,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)

        if hasattr(sample_graph, 'merchant_ids'):
            model.merchant_ids = sample_graph.merchant_ids
            model.merchant_global_map = merchant_global_map
        else:
            logging.warning("当前图数据中没有 merchant_ids 属性！")

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scaler = GradScaler()

        for epoch in range(1, epochs + 1):
            logging.info(f"{month} - Epoch {epoch}/{epochs} 开始训练...")
            train_loss = train_supervised(model, loader, optimizer, device, scaler, merchant_labels, criterion)
            logging.info(f"{month} - Epoch {epoch}: Train Loss = {train_loss:.4f}")
            torch.cuda.empty_cache()

        # 模型评估：使用 sigmoid 得到预测值并计算简单准确率
        model.eval()
        monthly_predictions = {}
        total, correct = 0, 0
        with torch.no_grad():
            for data in graphs:
                data = data.to(device)
                _, logits = model(data.x_dict, data.edge_index_dict)
                probs = torch.sigmoid(logits).view(-1)
                preds = (probs > 0.5).long()
                if hasattr(data, 'merchant_ids'):
                    for mid, pred in zip(data.merchant_ids, preds):
                        mid = str(mid)
                        monthly_predictions[mid] = int(pred.item())
                        target = int(merchant_labels.get(mid, 0))
                        if pred.item() == target:
                            correct += 1
                        total += 1
                else:
                    for idx, pred in enumerate(preds):
                        key = f"merchant_{idx}"
                        monthly_predictions[key] = int(pred.item())
                        target = int(merchant_labels.get(key, 0))
                        if pred.item() == target:
                            correct += 1
                        total += 1
            if total > 0:
                acc = correct / total
                logging.info(f"{month} - Evaluation accuracy: {acc:.4f}")
            else:
                logging.warning("评估时没有找到 merchant_ids 信息，无法计算准确率。")

        # 保存模型参数（checkpoint）
        checkpoint_dir = os.path.join("checkpoints", month)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_file = os.path.join(checkpoint_dir, f"supervised_model_{month}.pth")
        try:
            torch.save(model.state_dict(), model_file)
            logging.info(f"{month} - 模型已保存到: {model_file}")
        except Exception as e:
            logging.error(f"保存模型失败：{e}")

if __name__ == "__main__":
    main()