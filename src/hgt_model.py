import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData


# -------------------------------
# HGT 模型定义
# -------------------------------
class HGTModel(nn.Module):
    def __init__(self, in_channels_dict, hidden_channels, out_channels, metadata, num_heads=4, num_layers=2):
        """
        Args:
            in_channels_dict (dict): 每种节点的输入特征维度，例如 {'merchant': 20, 'goods': 50}
            hidden_channels (int): 隐藏层特征维度
            out_channels (int): 输出嵌入的目标维度（例如 128）
            metadata (tuple): HeteroData 对象的元信息，如 (node_types, edge_types)
            num_heads (int): 注意力头数
            num_layers (int): HGT 层数
        """
        super(HGTModel, self).__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.metadata = metadata

        # 对每种节点进行线性映射，将输入特征映射到 hidden_channels*num_heads 维
        self.lin_dict = nn.ModuleDict({
            node_type: nn.Linear(in_channels, hidden_channels * num_heads)
            for node_type, in_channels in in_channels_dict.items()
        })

        # 构造多个 HGTConv 层
        self.hgt_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hgt_layers.append(
                HGTConv(
                    in_channels=hidden_channels * num_heads,
                    out_channels=hidden_channels,
                    metadata=metadata,
                    heads=num_heads
                )
            )

        # 输出层：将多头融合后的特征映射到最终的 out_channels
        self.out_lin_dict = nn.ModuleDict({
            node_type: nn.Linear(hidden_channels * num_heads, out_channels)
            for node_type in in_channels_dict.keys()
        })

    def forward(self, data: HeteroData):
        # 初始对每类节点做线性变换，得到形状 (num_nodes, hidden_channels*num_heads)
        x_dict = {}
        for node_type, x in data.x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x)

        # 依次通过多个 HGTConv 层，每层后采用 ELU 激活函数
        for conv in self.hgt_layers:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {nt: F.elu(x) for nt, x in x_dict.items()}

        # 输出层映射，得到最终嵌入
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.out_lin_dict[node_type](x)
        return out_dict


# -------------------------------
# 图数据加载
# -------------------------------
def load_graph(graph_path):
    """
    加载保存的 PyG 异构图（HeteroData 对象）
    """
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    return torch.load(graph_path)


# -------------------------------
# 遍历图文件并提取嵌入
# -------------------------------
def extract_embeddings_from_all_graphs(graph_dir, hidden_channels=64, out_channels=128, num_heads=4, num_layers=2):
    """
    遍历 graph_dir 中所有 .pt 图文件，利用 HGT 模型提取嵌入，返回一个字典：
         { time_str: {'merchant': tensor, 'goods': tensor} }
    """
    embedding_dict = {}
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pt')])

    for graph_file in graph_files:
        graph_path = os.path.join(graph_dir, graph_file)
        logging.info(f"Processing graph: {graph_path}")
        data = load_graph(graph_path)

        # 获取图元数据：例如 (['merchant', 'goods'], [('merchant', 'ordered', 'goods')])
        metadata = data.metadata()

        # 获取每种节点的输入特征维度
        in_channels_dict = {}
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x'):
                in_channels_dict[node_type] = data[node_type].x.size(1)
            else:
                raise ValueError(f"节点类型 '{node_type}' 缺少 x 属性，请检查图构建代码。")
        logging.info(f"In_channels for {graph_file}: {in_channels_dict}")

        # 初始化 HGT 模型并提取嵌入
        model = HGTModel(in_channels_dict, hidden_channels, out_channels, metadata,
                         num_heads=num_heads, num_layers=num_layers)
        model.eval()
        with torch.no_grad():
            embeddings = model(data)

        # 通过文件名提取时间信息，例如 "2021-11"
        time_str = graph_file.replace('hetero_graph_', '').replace('.pt', '')
        embedding_dict[time_str] = embeddings

        for node_type, emb in embeddings.items():
            logging.info(f"Graph [{time_str}] - 节点类型 '{node_type}' 嵌入形状: {emb.shape}")

    return embedding_dict


def main():
    logging.basicConfig(level=logging.INFO)

    # 指定存放 PyG 图的目录（例如存放了 hetero_graph_2021-11.pt 到 hetero_graph_2023-11.pt 文件）
    graph_dir = "data/pyg_graphs"

    # 提取所有图的嵌入
    embedding_dict = extract_embeddings_from_all_graphs(graph_dir)

    # 示例：保存所有时间截面中 merchant 节点的嵌入到一个文件中
    merchant_embeddings = {time_str: emb_dict['merchant'] for time_str, emb_dict in embedding_dict.items()
                           if 'merchant' in emb_dict}
    torch.save(merchant_embeddings, "all_merchant_embeddings.pt")
    logging.info("保存所有时间截面的 merchant 节点嵌入到 'all_merchant_embeddings.pt'")


if __name__ == "__main__":
    main()