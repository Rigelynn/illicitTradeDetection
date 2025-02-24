#!/usr/bin/env python
# train_unsupervised_hgt_with_prompt.py

import os

# 指定使用 GPU 5 或 GPU 6，根据需求修改
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch_geometric.loader import DataLoader  # 若有条件，可改用 ClusterLoader/NeighborSampler
from torch_geometric.nn import HGTConv
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def unsupervised_loss_function(embeddings, data):
    """
    占位的无监督损失函数，这里采用简单的 L2 正则作为示例。
    实际中建议替换为对比学习损失（InfoNCE、DGI 等），使得模型学到更有区分性的嵌入。
    """
    loss = torch.mean(embeddings ** 2)
    return loss


class UnsupervisedHGTWithPrompt(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, embed_size, metadata,
                 prompt_length, llm_embedding_dim, num_heads=4, num_layers=2):
        """
        Args:
            in_channels_dict (dict): 每种节点的输入特征维度，例如 {'merchant': 13, 'goods': 430, ...}
            hidden_channels (int): HGTConv 层输出的隐藏特征维度（例如 128）
            embed_size (int): HGT 最终输出的嵌入维度
            metadata (tuple): 图的元信息，形态为 (node_types, edge_types)
            prompt_length (int): 生成连续 prompt tokens 的数量（例如 10 个 token）
            llm_embedding_dim (int): 下游 LLM 词嵌入的维度（例如 768）
            num_heads (int): HGT 的注意力头数量
            num_layers (int): HGTConv 层数
        """
        super(UnsupervisedHGTWithPrompt, self).__init__()
        self.num_layers = num_layers
        self.metadata = metadata

        # 构建 HGTConv 层，每一层输出特征维度均为 hidden_channels（使用默认 add 聚合方式）
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # 第一层使用原始特征维度字典
                conv_in_channels = in_channels_dict
            else:
                # 后续层输入均为 hidden_channels
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

        # 最终嵌入层：输入维度为 hidden_channels（128）
        self.embed_layer = nn.Linear(hidden_channels, embed_size)

        # Prompt reprogrammer：将 HGT 嵌入映射为连续 prompt tokens
        self.prompt_length = prompt_length
        self.llm_embedding_dim = llm_embedding_dim
        # 输出维度为 prompt_length * llm_embedding_dim，之后再 reshape 成 (num_nodes, prompt_length, llm_embedding_dim)
        self.prompt_reprogrammer = nn.Linear(embed_size, prompt_length * llm_embedding_dim)

        # 针对 "merchant" 节点，由于其原始特征维度为 13，与 hidden_channels（128）不匹配，
        # 我们构造一个投影层，把它从 13 映射到 128，
        # 该投影将在每层 HGTConv 后应用（如果 "merchant" 节点没有收到消息）。
        self.proj_first = nn.ModuleDict()
        if "merchant" in in_channels_dict and in_channels_dict["merchant"] != hidden_channels:
            self.proj_first["merchant"] = nn.Linear(in_channels_dict["merchant"], hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        前向传播：
          参数：
            x_dict: dict 格式的节点特征，例如 {'merchant': tensor, 'goods': tensor, ...}
            edge_index_dict: dict 格式的边索引信息
          返回：
            embeddings: 商户节点嵌入，形状 (num_merchants, embed_size)
            prompt_tokens: 生成的连续 prompt tokens，形状 (num_merchants, prompt_length, llm_embedding_dim)
        """
        # 在进入多层传播前备份原始表示
        x_dict_backup = x_dict.copy()

        for layer in self.layers:
            if isinstance(layer, HGTConv):
                out = layer(x_dict, edge_index_dict)
                # 针对 "merchant" 节点：如果该层没有更新 "merchant"（即没有消息传入），则采用上一层表示。
                # 如果上一层的表示维度不是 hidden_channels，则通过投影转换。
                if "merchant" not in out:
                    merchant_feat = x_dict["merchant"]
                    if merchant_feat.shape[1] != self.embed_layer.in_features:  # self.embed_layer.in_features==hidden_channels
                        merchant_feat = self.proj_first["merchant"](merchant_feat)
                    out["merchant"] = merchant_feat
                x_dict = {k: torch.relu(v) for k, v in out.items()}
            else:
                x_dict = {k: layer(v) for k, v in x_dict.items()}

        # 最终获得 "merchant" 节点表示（预计应为 hidden_channels 维度）
        merchant_features = x_dict.get("merchant", x_dict_backup.get("merchant"))
        if merchant_features.shape[1] != self.embed_layer.in_features:
            raise KeyError(f"'merchant' 节点特征维度为 {merchant_features.shape[1]}，而期望为 {self.embed_layer.in_features}")

        embeddings = self.embed_layer(merchant_features)  # (num_merchants, embed_size)

        # 生成连续 prompt tokens
        prompt_tokens = self.prompt_reprogrammer(embeddings)  # (num_merchants, prompt_length * llm_embedding_dim)
        prompt_tokens = prompt_tokens.view(-1, self.prompt_length, self.llm_embedding_dim)

        return embeddings, prompt_tokens


def load_graph(graph_path):
    """
    加载保存的 HeteroData 图。
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"图文件不存在: {graph_path}")
    return torch.load(graph_path)


def prepare_unsupervised_dataset(graph_dir):
    """
    遍历图文件目录，将所有图加载成列表，供无监督训练使用。
    """
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pt')])
    data_list = []
    for gf in graph_files:
        graph_path = os.path.join(graph_dir, gf)
        data = load_graph(graph_path)
        data_list.append(data)
    return data_list


def train_unsupervised(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc="Training")
    for data in progress_bar:
        try:
            data = data.to(device)
        except Exception as e:
            print(f"数据转移到设备时出错: {e}")
            continue

        optimizer.zero_grad()
        try:
            with autocast():
                embeddings, prompt_tokens = model(data.x_dict, data.edge_index_dict)
                loss = unsupervised_loss_function(embeddings, data)
        except Exception as e:
            print(f"前向传播时出错: {e}")
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.num_graphs
        progress_bar.set_postfix({"Loss": loss.item()})
    return total_loss / len(loader)


def evaluate_unsupervised(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Evaluating")
        for data in progress_bar:
            try:
                data = data.to(device)
            except Exception as e:
                print(f"数据转移到设备时出错: {e}")
                continue

            try:
                with autocast():
                    embeddings, prompt_tokens = model(data.x_dict, data.edge_index_dict)
                    loss = unsupervised_loss_function(embeddings, data)
            except Exception as e:
                print(f"前向传播时出错: {e}")
                continue

            total_loss += loss.item() * data.num_graphs
            progress_bar.set_postfix({"Loss": loss.item()})
    return total_loss / len(loader)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 指定保存 HeteroData 图的目录，请确保此目录下存在 .pt 文件
    graph_dir = "/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/pyg_graphs"
    logging.info("准备无监督训练数据集...")
    data_list = prepare_unsupervised_dataset(graph_dir)
    if not data_list:
        logging.error("没有可用的数据进行训练，请检查图文件是否存在。")
        return

    # 由于每张图非常大，建议每次只加载一张图：batch_size 设置为 1
    batch_size = 1
    loader = DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    # 从任一图中提取图元信息和各类型节点的输入特征维度
    sample_graph = data_list[0]
    metadata = sample_graph.metadata()  # 返回 (node_types, edge_types)
    in_channels_dict = {nt: sample_graph[nt].x.size(1) for nt in sample_graph.node_types}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数设置
    hidden_channels = 128
    embed_size = 256
    num_heads = 8
    num_layers = 4

    # 针对 prompt reprogramming 的参数
    prompt_length = 10
    llm_embedding_dim = 768

    model = UnsupervisedHGTWithPrompt(
        in_channels_dict,
        hidden_channels,
        embed_size,
        metadata,
        prompt_length,
        llm_embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    epochs = 50
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        logging.info(f"开始 Epoch {epoch}/{epochs} 的训练...")
        train_loss = train_unsupervised(model, loader, optimizer, device, scaler)
        logging.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        logging.info(f"开始 Epoch {epoch}/{epochs} 的验证...")
        val_loss = evaluate_unsupervised(model, loader, device)
        logging.info(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

        # 每个 epoch 结束后清理显存
        torch.cuda.empty_cache()

        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"已保存模型检查点: {checkpoint_path}")

    torch.save(model.state_dict(), "unsupervised_hgt_with_prompt_final.pth")
    logging.info("无监督 HGT 带 prompt 模块训练完成，模型已保存为 'unsupervised_hgt_with_prompt_final.pth'")


if __name__ == "__main__":
    main()