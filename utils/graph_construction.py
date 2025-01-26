import dgl
import pandas as pd
import numpy as np
import torch


def construct_heterogeneous_graph(df, epsilon=0.8):
    """
    从清洗后的订单数据构建异构图。

    参数：
        df (pd.DataFrame): 清洗后的订单数据。
        epsilon (float): 订单相似度阈值。

    返回：
        dgl.DGLHeteroGraph：构建的异构图。
        merchants (pd.DataFrame)：商户信息。
        tobaccos (pd.DataFrame)：烟草类型信息。
    """
    # 定义节点类型
    merchant_type = 'merchant'
    tobacco_type = 'tobacco'

    # 定义边类型
    orders_rel = ('merchant', 'orders', 'tobacco')
    geos_rel = ('merchant', 'geos', 'merchant')
    similar_orders_rel = ('merchant', 'similar_orders', 'merchant')

    # 获取唯一的商户和烟草类型
    merchants = df[['COMID', 'LICEN_NO']].drop_duplicates().reset_index(drop=True)
    tobaccos = df['GOODSNAME'].unique()

    # 创建映射关系
    merchant_map = {comid: idx for idx, comid in enumerate(merchants['COMID'])}
    tobacco_map = {name: idx for idx, name in enumerate(tobaccos)}

    # 初始化边的数据
    graph_data = {}

    # 构建订单关系边
    src_orders = df['COMID'].map(merchant_map).values
    dst_orders = df['GOODSNAME'].map(tobacco_map).values
    graph_data[orders_rel] = (src_orders, dst_orders, df['QTY'].values)

    # 构建地理位置关系边
    geo_groups = merchants.groupby('LICEN_NO')['COMID'].apply(list)
    src_geos = []
    dst_geos = []
    for group in geo_groups:
        if len(group) > 1:
            merchant_indices = [merchant_map[comid] for comid in group]
            for i in merchant_indices:
                for j in merchant_indices:
                    if i != j:
                        src_geos.append(i)
                        dst_geos.append(j)
    graph_data[geos_rel] = (src_geos, dst_geos, np.ones(len(src_geos)))

    # 构建订单相似度关系边
    # 创建订单模式向量（每个商户对每种烟草的总订单量）
    order_pivot = df.pivot_table(index='COMID', columns='GOODSNAME', values='QTY', aggfunc='sum', fill_value=0)
    ordering_vectors = torch.tensor(order_pivot.values, dtype=torch.float)

    # 计算余弦相似度
    similarity_matrix = torch.nn.functional.cosine_similarity(ordering_vectors.unsqueeze(1),
                                                              ordering_vectors.unsqueeze(0), dim=2)

    # 生成相似订单边（相似度 > epsilon，且避免自环）
    similar_pairs = torch.nonzero((similarity_matrix > epsilon) & (similarity_matrix < 1.0))
    src_similar = similar_pairs[:, 0].tolist()
    dst_similar = similar_pairs[:, 1].tolist()
    graph_data[similar_orders_rel] = (src_similar, dst_similar, np.ones(len(src_similar)))

    # 创建 DGL 异构图
    hg = dgl.heterograph({
        orders_rel: (graph_data[orders_rel][0], graph_data[orders_rel][1]),
        geos_rel: (graph_data[geos_rel][0], graph_data[geos_rel][1]),
        similar_orders_rel: (graph_data[similar_orders_rel][0], graph_data[similar_orders_rel][1])
    })

    # 分配边特征
    for rel in graph_data:
        hg.edges[rel].data['edge_attr'] = torch.tensor(graph_data[rel][2], dtype=torch.float)

    # 分配节点特征
    # 商户节点特征：基础信息（MerchantID 和 LicenseID 的 one-hot 编码）
    # 假设我们只是简单地使用 one-hot 编码表示商户
    merchant_features = torch.eye(len(merchants))
    hg.nodes['merchant'].data['x_b'] = merchant_features

    # 订单信息嵌入（每个商户对每种烟草的订单量向量）
    order_embed = torch.tensor(order_pivot.values, dtype=torch.float)
    hg.nodes['merchant'].data['x_a'] = order_embed

    # 合并节点特征
    hg.nodes['merchant'].data['x'] = torch.cat([hg.nodes['merchant'].data['x_b'], hg.nodes['merchant'].data['x_a']],
                                               dim=1)

    # 烟草类型节点特征：one-hot 编码
    tobacco_features = torch.eye(len(tobaccos))
    hg.nodes['tobacco'].data['x'] = tobacco_features

    return hg, merchants, tobaccos


def main():
    # 加载清洗后的订单数据
    df = pd.read_csv("data/processed/orders_cleaned.csv")

    # 构建异构图
    print("构建异构图...")
    hg, merchants, tobaccos = construct_heterogeneous_graph(df, epsilon=0.8)

    # 保存图结构
    torch.save(hg, "data/processed/hetero_graph.pt")
    print("异构图已保存到 data/processed/hetero_graph.pt")

    # 保存商户和烟草类型映射
    merchants[['COMID', 'LICEN_NO']].to_csv("data/processed/merchants.csv", index=False)
    pd.Series(tobaccos).to_csv("data/processed/tobaccos.csv", index=False)
    print("商户和烟草类型映射已保存。")


if __name__ == "__main__":
    main()