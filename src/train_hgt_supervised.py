#!/usr/bin/env python
# train_hgt_supervised.py

import os
import sys
import logging
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import HGTConv
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import process_labels  # 请确保 process_labels.py 与本文件在同一目录下


def load_merchant_id_map(processed_dir):
    """
    加载存储的 merchant_id_map（映射 LICEN_NO 到节点索引）。
    """
    try:
        with open(os.path.join(processed_dir, 'merchant_id_map.pkl'), 'rb') as f:
            merchant_id_map = pickle.load(f)
        return merchant_id_map
    except Exception as e:
        logging.error(f"加载 merchant_id_map 失败: {e}")
        raise


def get_merchant_order_list(merchant_id_map):
    """
    根据 merchant_id_map（LICEN_NO -> index）生成一个列表，
    列表下标对应图中 merchant 节点的索引，元素为对应的 LICEN_NO。
    """
    # 建立逆映射： index -> LICEN_NO
    reverse_map = {v: k for k, v in merchant_id_map.items()}
    num_merchants = len(merchant_id_map)
    merchant_order_list = [reverse_map[i] for i in range(num_merchants)]
    return merchant_order_list


def load_graph(graph_path):
    """
    加载保存的 PyG HeteroData 图对象。
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"图文件 {graph_path} 不存在")
    return torch.load(graph_path)


def prepare_supervised_dataset(graph_dir):
    """
    遍历 graph_dir 中所有 .pt 格式的图文件，将他们加载为列表。
    """
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pt')])
    data_list = []
    for gf in graph_files:
        graph_path = os.path.join(graph_dir, gf)
        data = load_graph(graph_path)
        data_list.append(data)
    return data_list


# -------------- 定义 HGT 模型与监督任务模块 --------------

class UnsupervisedHGTWithPrompt(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, embed_size, metadata,
                 prompt_length, llm_embedding_dim, num_heads=4, num_layers=2):
        """
        该模块与前期无标签训练中相同，用于提取 merchant 节点嵌入，并生成连续的 prompt tokens。
        """
        super(UnsupervisedHGTWithPrompt, self).__init__()
        self.num_layers = num_layers
        self.metadata = metadata
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            conv_in_channels = in_channels_dict if i == 0 else {nt: hidden_channels for nt in metadata[0]}
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
        self.prompt_length = prompt_length
        self.llm_embedding_dim = llm_embedding_dim
        self.prompt_reprogrammer = nn.Linear(embed_size, prompt_length * llm_embedding_dim)
        # 针对 "merchant" 节点，如果特征维度不匹配，则构造投影层
        self.proj_first = nn.ModuleDict()
        if "merchant" in in_channels_dict and in_channels_dict["merchant"] != hidden_channels:
            self.proj_first["merchant"] = nn.Linear(in_channels_dict["merchant"], hidden_channels)

    def forward(self, x_dict, edge_index_dict):
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
            raise KeyError(f"'merchant' 节点特征维度为 {merchant_features.shape[1]}，期望 {self.embed_layer.in_features}")
        embeddings = self.embed_layer(merchant_features)  # (num_merchants, embed_size)
        prompt_tokens = self.prompt_reprogrammer(embeddings)
        prompt_tokens = prompt_tokens.view(-1, self.prompt_length, self.llm_embedding_dim)
        return embeddings, prompt_tokens


class SupervisedHGTWithLLMTask(nn.Module):
    def __init__(self, hgt_model, embed_size, llm_embedding_dim, mlp_hidden_dim, num_classes=1):
        """
        利用 HGT 提取的商户嵌入和 prompt tokens，
        融合后通过一个 MLP 层进行下游预测（例如二分类：违规 or 非违规）
        """
        super(SupervisedHGTWithLLMTask, self).__init__()
        self.hgt_model = hgt_model
        self.mlp = nn.Sequential(
            nn.Linear(embed_size + llm_embedding_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_classes)
        )

    def forward(self, x_dict, edge_index_dict):
        merchant_embedding, prompt_tokens = self.hgt_model(x_dict, edge_index_dict)
        # 对 prompt tokens 进行平均池化
        aggregated_prompt = torch.mean(prompt_tokens, dim=1)  # (num_merchants, llm_embedding_dim)
        combined_feature = torch.cat([merchant_embedding, aggregated_prompt], dim=1)
        logits = self.mlp(combined_feature)  # (num_merchants, num_classes)
        return logits


# -------------- 训练/验证函数 --------------

def train_supervised(model, loader, optimizer, device, scaler, merchant_label_dict, merchant_order_list):
    model.train()
    total_loss = 0
    bce_loss = nn.BCEWithLogitsLoss()
    progress_bar = tqdm(loader, desc="Supervised Training")
    for data in progress_bar:
        try:
            data = data.to(device)
        except Exception as e:
            print(f"数据转移到设备时出错: {e}")
            continue

        # 根据 merchant_order_list 构造标签，
        # merchant_order_list 的顺序必须与图中 merchant 节点一致。
        label_list = [1.0 if merchant in merchant_label_dict else 0.0 for merchant in merchant_order_list]
        labels = torch.tensor(label_list, dtype=torch.float, device=device).unsqueeze(1)

        optimizer.zero_grad()
        try:
            with autocast():
                logits = model(data.x_dict, data.edge_index_dict)  # (num_merchants, 1)
                loss = bce_loss(logits, labels)
        except Exception as e:
            print(f"前向传播出错: {e}")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.num_graphs
        progress_bar.set_postfix({"Loss": loss.item()})
    return total_loss / len(loader)


def evaluate_supervised(model, loader, device, merchant_label_dict, merchant_order_list):
    model.eval()
    total_loss = 0
    bce_loss = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Supervised Evaluating")
        for data in progress_bar:
            try:
                data = data.to(device)
            except Exception as e:
                print(f"数据转移到设备时出错: {e}")
                continue

            label_list = [1.0 if merchant in merchant_label_dict else 0.0 for merchant in merchant_order_list]
            labels = torch.tensor(label_list, dtype=torch.float, device=device).unsqueeze(1)

            try:
                with autocast():
                    logits = model(data.x_dict, data.edge_index_dict)
                    loss = bce_loss(logits, labels)
            except Exception as e:
                print(f"前向传播出错: {e}")
                continue

            total_loss += loss.item() * data.num_graphs
            progress_bar.set_postfix({"Loss": loss.item()})
    return total_loss / len(loader)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    processed_dir = "../data/processed/"
    graph_dir = os.path.join(processed_dir, "pyg_graphs")
    logging.info("加载有标签训练数据集...")
    data_list = prepare_supervised_dataset(graph_dir)
    if not data_list:
        logging.error("没有发现图数据，请检查目录。")
        sys.exit(1)

    batch_size = 1  # 每个图通常较大，可设置 batch_size=1
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # 从任一图数据中提取图元信息和各节点的输入特征维度
    sample_graph = data_list[0]
    metadata = sample_graph.metadata()  # (node_types, edge_types)
    in_channels_dict = {nt: sample_graph[nt].x.size(1) for nt in sample_graph.node_types}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数
    hidden_channels = 128
    embed_size = 256
    num_heads = 8
    num_layers = 4
    prompt_length = 10
    llm_embedding_dim = 768

    base_hgt = UnsupervisedHGTWithPrompt(in_channels_dict, hidden_channels, embed_size, metadata,
                                           prompt_length, llm_embedding_dim,
                                           num_heads=num_heads, num_layers=num_layers).to(device)
    mlp_hidden_dim = 128
    supervised_model = SupervisedHGTWithLLMTask(base_hgt, embed_size, llm_embedding_dim, mlp_hidden_dim, num_classes=1).to(device)

    optimizer = optim.Adam(supervised_model.parameters(), lr=0.001)
    scaler = GradScaler()

    # 加载标签数据
    label_file = "../data/raw/illicitInfo.xlsx"
    _, merchant_label_dict = process_labels.process_label_data(label_file)

    # 加载 merchant_id_map 并构造 merchant_order_list，
    # merchant_order_list 中的每个元素与图中 merchant 节点的顺序一致，元素为对应的 LICEN_NO
    merchant_id_map = load_merchant_id_map(processed_dir)
    merchant_order_list = get_merchant_order_list(merchant_id_map)

    epochs = 50
    checkpoint_dir = "checkpoints_supervised"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        logging.info(f"Epoch {epoch}/{epochs}: 开始训练...")
        train_loss = train_supervised(supervised_model, loader, optimizer, device, scaler,
                                      merchant_label_dict, merchant_order_list)
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        logging.info(f"Epoch {epoch}/{epochs}: 开始验证...")
        val_loss = evaluate_supervised(supervised_model, loader, device,
                                       merchant_label_dict, merchant_order_list)
        logging.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

        torch.cuda.empty_cache()
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(supervised_model.state_dict(), checkpoint_path)
        logging.info(f"保存检查点: {checkpoint_path}")

    torch.save(supervised_model.state_dict(), "supervised_hgt_with_llm_final.pth")
    logging.info("有标签训练完成，模型保存为 'supervised_hgt_with_llm_final.pth'.")


if __name__ == "__main__":
    main()