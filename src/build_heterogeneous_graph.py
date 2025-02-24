import torch
import dgl
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def build_heterogeneous_graph(df_snapshot, merchant_id_map_global, goods_id_map_global):
    """
    构建给定时间截面的异构图。

    参数：
    - df_snapshot: 当期时间截面的订单数据
    - merchant_id_map_global: 全局商户ID映射
    - goods_id_map_global: 全局商品ID映射

    返回：
    - hetero_graph: DGL异构图
    """
    NODE_TYPES = ['merchant', 'goods']
    EDGE_TYPES = ['ordered', 'geographical']

    # 获取唯一商户和商品，确保使用全局ID映射
    merchants = df_snapshot['LICEN_NO'].unique()
    goods = df_snapshot['GOODSNAME'].unique()

    # 使用全局ID映射，如果商户或商品不在全局映射中，则忽略
    merchants = [m for m in merchants if m in merchant_id_map_global]
    goods = [g for g in goods if g in goods_id_map_global]

    if len(merchants) == 0 or len(goods) == 0:
        logging.warning("当前时间截面没有有效的商户或商品数据。")
        return None

    # 准备边数据

    ### 1. ordered edges: merchant -> goods
    ordering_src = df_snapshot['LICEN_NO'].map(merchant_id_map_global).tolist()
    ordering_dst = df_snapshot['GOODSNAME'].map(goods_id_map_global).tolist()

    # 添加边特征
    ordering_time = df_snapshot['order_time'].tolist()
    ordering_amount = df_snapshot['AMT'].tolist()  # 假设有销售金额列

    # 将时间转换为数值，例如 UNIX 时间戳
    ordering_timestamp = [datetime.timestamp(t) for t in ordering_time]
    ordering_amount_tensor = torch.tensor(ordering_amount, dtype=torch.float32)
    ordering_timestamp_tensor = torch.tensor(ordering_timestamp, dtype=torch.float32)

    # 创建异构边
    src_ordered = torch.tensor(ordering_src, dtype=torch.int64)
    dst_ordered = torch.tensor(ordering_dst, dtype=torch.int64)

    # 去重或根据需要进行处理，这里保留所有订单边，允许多边
    ordered_graph = dgl.graph((src_ordered, dst_ordered), num_nodes=len(merchant_id_map_global))
    ordered_graph.edata['timestamp'] = ordering_timestamp_tensor
    ordered_graph.edata['amount'] = ordering_amount_tensor

    ### 2. geographical edges: merchant <-> merchant
    geographical_edges = []
    merchant_geo = df_snapshot[['LICEN_NO', 'PROVINCE', 'CITY', 'DISTRICT']].drop_duplicates()
    merchant_geo = merchant_geo.set_index('LICEN_NO')
    merchants_list = merchant_geo.index.tolist()

    # 构建地理关系（同省市区）
    for i in range(len(merchants_list)):
        for j in range(i + 1, len(merchants_list)):
            comid_i = merchants_list[i]
            comid_j = merchants_list[j]
            if (merchant_geo.loc[comid_i, 'PROVINCE'] == merchant_geo.loc[comid_j, 'PROVINCE'] and
                    merchant_geo.loc[comid_i, 'CITY'] == merchant_geo.loc[comid_j, 'CITY'] and
                    merchant_geo.loc[comid_i, 'DISTRICT'] == merchant_geo.loc[comid_j, 'DISTRICT']):
                geographical_edges.append((merchant_id_map_global[comid_i], merchant_id_map_global[comid_j]))
                geographical_edges.append((merchant_id_map_global[comid_j], merchant_id_map_global[comid_i]))

    if geographical_edges:
        src_geo = torch.tensor([edge[0] for edge in geographical_edges], dtype=torch.int64)
        dst_geo = torch.tensor([edge[1] for edge in geographical_edges], dtype=torch.int64)
        geographical_graph = dgl.graph((src_geo, dst_geo), num_nodes=len(merchant_id_map_global))
    else:
        geographical_graph = dgl.graph(([], []), num_nodes=len(merchant_id_map_global))

    # 构建边字典
    data_dict = {}
    data_dict[('merchant', 'ordered', 'goods')] = (ordered_graph.edges()[0], ordered_graph.edges()[1])
    data_dict[('merchant', 'geographical', 'merchant')] = (geographical_graph.edges()[0], geographical_graph.edges()[1])

    # 创建异构图
    hetero_graph = dgl.heterograph(data_dict)

    # 添加边特征
    hetero_graph.edges['ordered'].data['timestamp'] = ordering_timestamp_tensor
    hetero_graph.edges['ordered'].data['amount'] = ordering_amount_tensor
    # 若有地理特征，可在这里添加

    # 添加节点特征（如果有的话）
    # 此处假设商户节点特征包含时间特征
    num_features = 8  # BORN_YEAR, BORN_MONTH, BORN_DAY, BORN_WEEKDAY, CONFORM_YEAR, CONFORM_MONTH, CONFORM_DAY, CONFORM_WEEKDAY
    merchant_features_df = df_snapshot[['LICEN_NO', 'BORN_YEAR', 'BORN_MONTH', 'BORN_DAY', 'BORN_WEEKDAY',
                                        'CONFORM_YEAR', 'CONFORM_MONTH', 'CONFORM_DAY',
                                        'CONFORM_WEEKDAY']].drop_duplicates(subset=['LICEN_NO'])
    merchant_features_df = merchant_features_df.set_index('LICEN_NO')

    # 按商户顺序排列特征
    features = []
    for comid in merchants:
        if comid in merchant_features_df.index:
            feature = merchant_features_df.loc[comid].values.astype(float)
            features.append(feature)
        else:
            features.append([0.0] * num_features)  # 如果缺失，使用零向量

    features = torch.tensor(features, dtype=torch.float32)
    hetero_graph.nodes['merchant'].data['features'] = features

    # 商品节点特征（如果有的话）
    # 示例：可以为商品添加类别、价格等特征
    # 如果没有，可以跳过

    logging.info(f"已构建时间截面为 {df_snapshot['time_snapshot'].iloc[0]} 的异构图。")

    return hetero_graph