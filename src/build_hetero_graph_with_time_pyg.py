# build_hetero_graph_with_time_pyg.py

import torch
import pandas as pd
import logging
from torch_geometric.data import HeteroData

def build_pyg_hetero_graph(df_orders, merchant_id_map, goods_id_map, region_id_map, include_timestamp=False):
    """
    构建 PyG 格式的异构图（HeteroData 对象），包含：
      - 节点：'merchant' 和 'goods'
      - 边：('merchant', 'ordered', 'goods')
      - 节点特征：
          • 商户节点使用 LICEN_NO 的前六位作为行政区划代码的 One-Hot 编码，
          • 商品节点使用 One-Hot 编码
      - 边特征：订单金额，及可选的订单时间戳

    参数：
      df_orders        : pandas.DataFrame，订单数据，必须包含 'LICEN_NO','GOODSNAME','AMT','CONFORM_DATE'
      merchant_id_map  : dict，商户名称到唯一整数ID的映射（全局固定顺序）
      goods_id_map     : dict，商品名称到唯一整数ID的映射
      region_id_map    : dict，区域代码到唯一整数ID的映射（全局固定顺序）
      include_timestamp: bool，是否添加订单时间戳边特征

    返回：
      data: PyG 的 HeteroData 对象
    """
    # 过滤掉商户和商品信息缺失的行
    df_ordered = df_orders.dropna(subset=['LICEN_NO', 'GOODSNAME']).copy()

    # 映射商户和商品到整数ID
    df_ordered['merchant_id'] = df_ordered['LICEN_NO'].map(merchant_id_map).astype(int)
    df_ordered['goods_id'] = df_ordered['GOODSNAME'].map(goods_id_map).astype(int)

    # 创建 HeteroData 对象
    data = HeteroData()
    num_merchants = len(merchant_id_map)
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
        # 补全缺失的时间戳，避免报错
        timestamp = timestamp.fillna(pd.Timestamp(0))
        timestamp_tensor = torch.tensor([ts.timestamp() for ts in timestamp], dtype=torch.float32)
        data['merchant', 'ordered', 'goods'].timestamp = timestamp_tensor

    # ---------- 添加商户节点特征：全局 REGIONS 的 One-Hot 编码 ----------
    num_regions = len(region_id_map)

    if 'LICEN_NO' in df_orders.columns:
        # 使用 drop_duplicates 保证每个商户只保留一个记录，避免重复的列名问题
        merchant_region = df_orders[['LICEN_NO']].drop_duplicates().copy()
        merchant_region['REGION'] = merchant_region['LICEN_NO'].astype(str).str[:6]

        merchant_feature_list = []
        # 按照全局 mapping 的顺序生成特征，保证后续各时间截面顺序一致
        for merchant in sorted(merchant_id_map.keys(), key=lambda k: merchant_id_map[k]):
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
    else:
        logging.warning("缺少 'LICEN_NO' 列，商户节点没有地理特征。")

    # 添加商品节点特征：One-Hot 编码
    goods_onehot = torch.eye(num_goods, dtype=torch.float32)
    data['goods'].x = goods_onehot

    logging.info(f"构建 PyG HeteroData 图：商户节点 {num_merchants}, 商品节点 {num_goods}, 边数 {edge_index.size(1)}")
    return data


def add_time_encoding(data, time_window_size=12):
    """
    可选：添加时间编码特征到节点或边，可以根据需求扩展。

    目前，此函数保留为空，供未来扩展使用。
    """
    pass  # 根据具体需求实现时间编码


if __name__ == "__main__":
    import logging
    import pickle

    logging.basicConfig(level=logging.INFO)
    """
    # 示例用法
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