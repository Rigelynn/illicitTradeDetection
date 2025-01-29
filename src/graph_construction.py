# src/graph_construction.py

import pandas as pd
import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import os


class HANLayer(nn.Module):
    """
    HAN (Heterogeneous Attention Network) Layer
    处理基于元路径的卷积和聚合。
    """

    def __init__(self, in_dim, out_dim, meta_paths):
        super(HANLayer, self).__init__()
        self.meta_paths = meta_paths
        self.conv = nn.ModuleDict({
            'orders': GraphConv(in_dim, out_dim),
            'geographical': GraphConv(in_dim, out_dim)
        })
        self.attn = nn.ModuleDict({
            mp: nn.Linear(out_dim, 1) for mp in meta_paths
        })

    def forward(self, g, h):
        path_convs = {}
        for mp in self.meta_paths:
            try:
                path_g = g.metapath(mp)
            except KeyError:
                print(f"Meta-path {mp} not found in the graph. Skipping.")
                continue
            conv = self.conv[mp[-1]](path_g, h[mp[0]])
            path_convs[mp] = conv

        # 聚合不同元路径的卷积结果
        h_new = {}
        for mp, conv_out in path_convs.items():
            if mp[0] not in h_new:
                h_new[mp[0]] = conv_out
            else:
                h_new[mp[0]] += conv_out
        return h_new


class HAN(nn.Module):
    """
    HAN (Heterogeneous Attention Network) Model
    堆叠多个HANLayer并聚合元路径信息。
    """

    def __init__(self, in_dim, hidden_dim, meta_paths, num_layers=2):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HANLayer(in_dim, hidden_dim, meta_paths))
            in_dim = hidden_dim  # 更新输入维度

        # Attention weights for different meta-paths
        self.attention = nn.Parameter(torch.Tensor(len(meta_paths), hidden_dim))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)

        # 聚合不同元路径的特征
        h_combined = {}
        for ntype in h:
            h_combined[ntype] = h[ntype]  # 简单聚合，可以进一步加入注意力机制
        return h_combined


def build_heterogeneous_graph(df_ordering, df_fraud, epsilon=0.8):
    """
    构建异构图，包括订购关系和地理关系的元路径
    """
    # 创建商户节点
    merchants = df_ordering['COMID'].unique()
    merchant_id_map = {comid: idx for idx, comid in enumerate(merchants)}

    # 创建商品节点
    goods = df_ordering['GOODSNAME'].unique()
    goods_id_map = {goods_name: idx + len(merchant_id_map) for idx, goods_name in enumerate(goods)}

    # 映射商户和商品ID
    df_ordering['merchant_id'] = df_ordering['COMID'].map(merchant_id_map)
    df_ordering['goods_id'] = df_ordering['GOODSNAME'].map(goods_id_map)

    # 创建订购关系边
    ordering_src = df_ordering['merchant_id'].tolist()
    ordering_dst = df_ordering['goods_id'].tolist()

    # 创建地理关系边（同一省市区的商户之间）
    df_ordering['PROVINCE'] = df_ordering['LICEN_NO'].astype(str).str[:2]
    df_ordering['CITY'] = df_ordering['LICEN_NO'].astype(str).str[2:4]
    df_ordering['DISTRICT'] = df_ordering['LICEN_NO'].astype(str).str[4:6]

    merchant_geo = df_ordering[['COMID', 'PROVINCE', 'CITY', 'DISTRICT']].drop_duplicates().set_index('COMID')

    # 根据地理信息创建地理关系元路径
    geographical_edges = []
    merchants_list = merchant_geo.index.tolist()
    num_merchants = len(merchants_list)
    for i in range(num_merchants):
        for j in range(i + 1, num_merchants):
            comid_i = merchants_list[i]
            comid_j = merchants_list[j]
            if (merchant_geo.loc[comid_i, 'PROVINCE'] == merchant_geo.loc[comid_j, 'PROVINCE'] and
                    merchant_geo.loc[comid_i, 'CITY'] == merchant_geo.loc[comid_j, 'CITY'] and
                    merchant_geo.loc[comid_i, 'DISTRICT'] == merchant_geo.loc[comid_j, 'DISTRICT']):
                geographical_edges.append((merchant_id_map[comid_i], merchant_id_map[comid_j]))
                geographical_edges.append((merchant_id_map[comid_j], merchant_id_map[comid_i]))

    # 构建异构图
    data_dict = {
        ('merchant', 'orders', 'goods'): (ordering_src, ordering_dst),
        ('merchant', 'geographical', 'merchant'): ([], [])
    }

    if geographical_edges:
        geo_src, geo_dst = zip(*geographical_edges)
        data_dict[('merchant', 'geographical', 'merchant')] = (list(geo_src), list(geo_dst))

    g = dgl.heterograph(data_dict)

    # 计算订购相似度并添加 'ordering_similar' 边类型
    # 使用余弦相似度计算商户之间的订购相似度
    order_features = df_ordering.groupby('merchant_id')['QTY'].sum().unstack(fill_value=0)
    similarity_matrix = cosine_similarity(order_features)

    ordering_sim_edges = []
    for i in range(num_merchants):
        for j in range(i + 1, num_merchants):
            sim = similarity_matrix[i, j]
            if sim > epsilon:
                ordering_sim_edges.append((i, j))
                ordering_sim_edges.append((j, i))

    if ordering_sim_edges:
        sim_src, sim_dst = zip(*ordering_sim_edges)
        g.add_edges(sim_src, sim_dst, etype='ordering_similar')

    return g, merchant_id_map, goods_id_map


def save_heterogeneous_graph(g, path='data/processed/hetero_graph.bin'):
    """
    保存异构图
    """
    dgl.save_graphs(path, [g])
    print(f"Heterogeneous graph saved to {path}")


def load_heterogeneous_graph(path='data/processed/hetero_graph.bin'):
    """
    加载异构图
    """
    graphs, _ = dgl.load_graphs(path)
    return graphs[0]


def main():
    processed_dir = '../data/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    # 加载预处理后的订单数据和违规标签
    ordering_processed_path = os.path.join(processed_dir, 'ordering_processed.csv')
    fraud_labels_path = os.path.join(processed_dir, 'fraud_labels_processed.csv')

    if not os.path.exists(ordering_processed_path):
        raise FileNotFoundError(f"预处理后的订单数据文件未找到: {ordering_processed_path}")
    if not os.path.exists(fraud_labels_path):
        raise FileNotFoundError(f"预处理后的违规标签文件未找到: {fraud_labels_path}")

    df_ordering = pd.read_csv(ordering_processed_path)
    df_fraud = pd.read_csv(fraud_labels_path)

    # 构建异构图
    g, merchant_id_map, goods_id_map = build_heterogeneous_graph(df_ordering, df_fraud, epsilon=0.8)

    # 保存异构图
    save_heterogeneous_graph(g, path=os.path.join(processed_dir, 'hetero_graph.bin'))

    # 打印图的基本信息
    print(f"Graph Information:")
    print(f"  Number of node types: {g.num_ntypes}")
    print(f"  Number of edge types: {g.num_etypes}")
    for ntype in g.ntypes:
        print(f"  Node type '{ntype}': {g.num_nodes(ntype)} nodes")
    for etype in g.etypes:
        print(f"  Edge type '{etype}': {g.num_edges(etype)} edges")


if __name__ == "__main__":
    main()