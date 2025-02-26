#!/usr/bin/env python
# build_hetero_graph_with_time_pyg.py

import torch
import pandas as pd
import logging
from torch_geometric.data import HeteroData

def build_pyg_hetero_graph(df_orders, merchant_id_map, goods_id_map, region_id_map, include_timestamp=False):
    """
    构建 PyG 格式的异构图（HeteroData 对象），仅包含当前时间截面中活跃的商户。
    """
    # 过滤掉商户和商品信息缺失的行
    df_ordered = df_orders.dropna(subset=['LICEN_NO', 'GOODSNAME']).copy()

    # 仅保留当前时间截面中活跃的商户
    active_merchants = df_ordered['LICEN_NO'].unique()
    active_merchants = [m for m in active_merchants if m in merchant_id_map]
    logging.info(f"当前时间截面中活跃的商户数量: {len(active_merchants)}")

    # 如果没有活跃商户，返回空
    if not active_merchants:
        logging.warning("当前时间截面中没有活跃的商户。")
        return None

    # 创建活跃商户的局部映射
    local_merchant_id_map = {m: merchant_id_map[m] for m in active_merchants}
    sorted_merchants = sorted(local_merchant_id_map.keys(), key=lambda k: local_merchant_id_map[k])

    # 将原始商户标识映射到全局整数ID
    df_ordered = df_ordered[df_ordered['LICEN_NO'].isin(active_merchants)]
    df_ordered['merchant_id'] = df_ordered['LICEN_NO'].map(merchant_id_map).astype(int)
    df_ordered['goods_id'] = df_ordered['GOODSNAME'].map(goods_id_map).astype(int)

    # 创建 HeteroData 对象
    data = HeteroData()
    num_merchants = len(active_merchants)
    num_goods = len(goods_id_map)
    data['merchant'].num_nodes = num_merchants
    data['goods'].num_nodes = num_goods

    # 设置边的连接 ('merchant', 'ordered', 'goods')
    edge_index = torch.tensor([df_ordered['merchant_id'].tolist(),
                               df_ordered['goods_id'].tolist()], dtype=torch.long)
    data['merchant', 'ordered', 'goods'].edge_index = edge_index

    # 添加边特征：订单金额
    amount = torch.tensor(df_ordered['AMT'].astype(float).tolist(), dtype=torch.float32)
    data['merchant', 'ordered', 'goods'].amount = amount

    # 可选：添加订单时间戳
    if include_timestamp:
        timestamp = pd.to_datetime(df_ordered['CONFORM_DATE'], errors='coerce')
        timestamp = timestamp.fillna(pd.Timestamp(0))
        timestamp_tensor = torch.tensor([ts.timestamp() for ts in timestamp], dtype=torch.float32)
        data['merchant', 'ordered', 'goods'].timestamp = timestamp_tensor

    # ---------- 添加商户节点特征：全局 REGIONS 的 One-Hot 编码 ----------
    num_regions = len(region_id_map)

    if 'LICEN_NO' in df_orders.columns:
        # 使用 drop_duplicates 保证每个商户只保留一个记录
        merchant_region = df_orders[['LICEN_NO']].drop_duplicates().copy()
        merchant_region['REGION'] = merchant_region['LICEN_NO'].astype(str).str[:6]

        merchant_feature_list = []
        # 按照当前时间截面的商户顺序生成特征，保证后续各时间截面顺序一致
        for merchant in sorted_merchants:
            # 如果该商户在当前数据中存在则取其 REGION，否则用 'unknown'
            if merchant in merchant_region['LICEN_NO'].values:
                region = merchant_region.loc[merchant_region['LICEN_NO'] == merchant, 'REGION'].iloc[0]
            else:
                region = 'unknown'
            # 初始化全零向量，然后设定对应区域位置为 1
            feature = [0.0] * num_regions
            if region in region_id_map:
                feature[region_id_map[region]] = 1.0
            merchant_feature_list.append(feature)

        merchant_features_tensor = torch.tensor(merchant_feature_list, dtype=torch.float32)
        data['merchant'].x = merchant_features_tensor

        # 保存当前时间截面活跃的商户 IDs
        data.merchant_ids = sorted_merchants
    else:
        logging.warning("缺少 'LICEN_NO' 列，商户节点没有地理特征。")

    # 添加商品节点特征：One-Hot 编码
    goods_onehot = torch.eye(num_goods, dtype=torch.float32)
    data['goods'].x = goods_onehot

    logging.info(f"构建 PyG HeteroData 图：商户节点 {num_merchants}, 商品节点 {num_goods}, 边数 {edge_index.size(1)}")
    return data


def add_time_encoding(data, time_window_size=12):
    """可选：添加时间编码特征到节点或边，供未来扩展使用。"""
    pass

if __name__ == "__main__":
    import logging
    import pickle

    logging.basicConfig(level=logging.INFO)
    """
    # 示例用法：
    df_orders = pd.read_csv('your_orders.csv', parse_dates=['CONFORM_DATE'])
    with open('merchant_id_map.pkl', 'rb') as f:
        merchant_id_map = pickle.load(f)
    with open('goods_id_map.pkl', 'rb') as f:
        goods_id_map = pickle.load(f)
    with open('region_id_map.pkl', 'rb') as f:
        region_id_map = pickle.load(f)
    hetero_graph = build_pyg_hetero_graph(df_orders, merchant_id_map, goods_id_map, region_id_map, include_timestamp=False)
    torch.save(hetero_graph, 'hetero_graph.pt')
    """