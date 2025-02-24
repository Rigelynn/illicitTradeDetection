# build_hetero_graph_with_time.py

import torch
import dgl
import pandas as pd
import logging

def build_heterogeneous_graph(df_orders, merchant_id_map, goods_id_map):
    """
    构建异构图的拓扑结构。
    """
    df_orders['LICEN_NO'] = df_orders['LICEN_NO'].astype(str).str.strip()
    df_orders['GOODSNAME'] = df_orders['GOODSNAME'].astype(str).str.strip()

    # 映射源和目的节点
    src = df_orders['LICEN_NO'].map(merchant_id_map)
    dst = df_orders['GOODSNAME'].map(goods_id_map)

    # 处理缺失映射
    df_ordered = df_orders.dropna(subset=['LICEN_NO', 'GOODSNAME'])
    src_ordered = df_ordered['LICEN_NO'].map(merchant_id_map).astype(int).tolist()
    dst_ordered = df_ordered['GOODSNAME'].map(goods_id_map).astype(int).tolist()

    data_dict = {
        ('merchant', 'ordered', 'goods'): (torch.tensor(src_ordered, dtype=torch.int64),
                                           torch.tensor(dst_ordered, dtype=torch.int64))
    }

    # 创建异构图
    try:
        hetero_graph = dgl.heterograph(
            data_dict,
            num_nodes_dict={'merchant': len(merchant_id_map), 'goods': len(goods_id_map)}
        )
        logging.info(f"成功创建异构图 - 商户节点: {hetero_graph.number_of_nodes('merchant')}, "
                     f"商品节点: {hetero_graph.number_of_nodes('goods')}, "
                     f"边数: {hetero_graph.number_of_edges()}")
    except Exception as e:
        logging.error(f"创建异构图失败：{e}")
        raise

    return hetero_graph

def add_edge_features(hetero_graph, df_orders, include_timestamp=False):
    """
    将边特征添加到已构建的异构图中。
    """
    # 边特征
    amount = torch.tensor(df_orders['AMT'].astype(float).tolist(), dtype=torch.float32)
    assert amount.shape[0] == hetero_graph.number_of_edges(('merchant', 'ordered', 'goods')), \
        "amount 特征的数量与边的数量不一致！"

    hetero_graph.edges[('merchant', 'ordered', 'goods')].data['amount'] = amount
    logging.info("成功添加 'amount' 边特征。")

    if include_timestamp:
        # 转换日期为 UNIX 时间戳
        timestamp = pd.to_datetime(df_orders['CONFORM_DATE'], errors='coerce')
        # 注意：astype(int) 可能会引发警告或错误，推荐使用 .view() 或 .timestamp()
        timestamp_tensor = torch.tensor([ts.timestamp() for ts in timestamp], dtype=torch.float32)
        assert timestamp_tensor.shape[0] == hetero_graph.number_of_edges(('merchant', 'ordered', 'goods')), \
            "timestamp 特征的数量与边的数量不一致！"
        hetero_graph.edges[('merchant', 'ordered', 'goods')].data['timestamp'] = timestamp_tensor
        logging.info("成功添加 'timestamp' 边特征。")

    return hetero_graph

def add_node_features(hetero_graph, df_orders, merchant_id_map):
    """
    将节点特征（如地理位置信息）添加到异构图中的商户节点。
    """
    # 提取地理信息并进行 One-Hot 编码
    merchant_geo = df_orders.groupby('LICEN_NO').agg(lambda x: x.mode()[0]).reset_index()
    geo_features = pd.get_dummies(merchant_geo['PROVINCE'], prefix='prov')
    merchant_features_df = pd.concat([merchant_geo[['LICEN_NO']], geo_features], axis=1)
    merchant_features_df['LICEN_NO'] = merchant_features_df['LICEN_NO'].astype(str)

    # 按照 merchant_id_map 的顺序构建特征列表
    merchant_feature_list = []
    for comid in sorted(merchant_id_map.keys(), key=lambda k: merchant_id_map[k]):
        if comid in merchant_features_df['LICEN_NO'].values:
            # 提取对应行
            feat = merchant_features_df.loc[merchant_features_df['LICEN_NO'] == comid, geo_features.columns].values
            if feat.size > 0:
                feat = feat[0].astype(float)
            else:
                feat = [0.0] * geo_features.shape[1]
        else:
            feat = [0.0] * geo_features.shape[1]
        merchant_feature_list.append(feat)

    merchant_features_tensor = torch.tensor(merchant_feature_list, dtype=torch.float32)

    # 断言检查
    assert merchant_features_tensor.shape[0] == hetero_graph.number_of_nodes('merchant'), \
        f"商户特征数量 ({merchant_features_tensor.shape[0]}) 不等于商户节点数量 ({hetero_graph.number_of_nodes('merchant')})"
    assert merchant_features_tensor.shape[1] == geo_features.shape[1], \
        f"商户特征维度 ({merchant_features_tensor.shape[1]}) 不等于地理特征列数 ({geo_features.shape[1]})"

    hetero_graph.nodes['merchant'].data['geo_features'] = merchant_features_tensor
    logging.info(f"成功添加商户节点地理特征 'geo_features'，形状: {merchant_features_tensor.shape}")

    return hetero_graph

def build_heterogeneous_graph_with_time(df_orders, merchant_id_map, goods_id_map, include_timestamp=False):
    """
    构建包含时间信息和节点特征的异构图。
    """
    hetero_graph = build_heterogeneous_graph(df_orders, merchant_id_map, goods_id_map)
    hetero_graph = add_edge_features(hetero_graph, df_orders, include_timestamp)
    hetero_graph = add_node_features(hetero_graph, df_orders, merchant_id_map)
    return hetero_graph

if __name__ == "__main__":
    import logging
    import pickle

    logging.basicConfig(level=logging.DEBUG)

    # 示例：加载数据、映射文件，然后调用构建函数
    # 请根据实际情况替换下列读取代码
    # df_orders = pd.read_csv('your_orders.csv', parse_dates=['CONFORM_DATE'])
    # with open('merchant_id_map.pkl', 'rb') as f:
    #     merchant_id_map = pickle.load(f)
    # with open('goods_id_map.pkl', 'rb') as f:
    #     goods_id_map = pickle.load(f)
    # hetero_graph = build_heterogeneous_graph_with_time(df_orders, merchant_id_map, goods_id_map, include_timestamp=False)