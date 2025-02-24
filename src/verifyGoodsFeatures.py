import os
import pickle
import torch
import dgl
import logging
import sys


def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def load_goods_id_map(goods_id_map_path):
    """加载 goods_id_map.pkl 文件"""
    try:
        with open(goods_id_map_path, 'rb') as f:
            goods_id_map = pickle.load(f)
        logging.info(f"成功加载 goods_id_map.pkl，共 {len(goods_id_map)} 个商品。")
        return goods_id_map
    except Exception as e:
        logging.error(f"加载 goods_id_map.pkl 失败：{e}")
        raise


def load_graph(graph_path):
    """加载 .dgl 图文件"""
    try:
        graphs, _ = dgl.load_graphs(graph_path)
        hetero_graph = graphs[0]
        logging.info(f"成功加载图文件：{graph_path}")
        return hetero_graph
    except Exception as e:
        logging.error(f"加载图文件 {graph_path} 失败：{e}")
        raise


def verify_goods_onehot(hetero_graph, goods_id_map, num_samples=5):
    """
    验证 'goods_onehot' 特征是否正确对应 goods_id_map。

    参数：
        hetero_graph: dgl 异构图
        goods_id_map: dict，商品名称到唯一整数 ID 的映射
        num_samples: int，验证的商品数量
    """
    if 'goods' not in hetero_graph.ntypes:
        logging.error("图中不存在 'goods' 节点类型。")
        return

    if 'goods_onehot' not in hetero_graph.nodes['goods'].data:
        logging.error("图中 'goods' 节点未包含 'goods_onehot' 特征。")
        return

    goods_onehot = hetero_graph.nodes['goods'].data['goods_onehot']

    # 创建 ID 到商品名称的反向映射
    id_to_goods = {v: k for k, v in goods_id_map.items()}

    num_goods = len(goods_id_map)

    # 获取所有商品节点的索引
    goods_node_ids = list(goods_id_map.values())

    if goods_onehot.shape != torch.Size([num_goods, num_goods]):
        logging.warning(f"'goods_onehot' 特征的形状为 {goods_onehot.shape}，预期为 ({num_goods}, {num_goods})。")

    # 随机选择 num_samples 个商品进行验证
    import random
    sample_ids = random.sample(goods_node_ids, min(num_samples, num_goods))

    for gid in sample_ids:
        goods_name = id_to_goods.get(gid, "未知商品")
        onehot_vector = goods_onehot[gid]

        # 确认 One-Hot 向量中只有一个位置为 1，其余为 0
        if not torch.isclose(onehot_vector.sum(), torch.tensor(1.0)):
            logging.error(f"商品 '{goods_name}' (ID: {gid}) 的 One-Hot 向量总和不为 1：{onehot_vector}")
            continue

        # 找到 One-Hot 向量中值为 1 的索引
        onehot_index = torch.argmax(onehot_vector).item()

        if onehot_index != gid:
            logging.error(
                f"商品 '{goods_name}' (ID: {gid}) 的 One-Hot 编码不正确。期望索引 {gid}，实际索引 {onehot_index}。")
        else:
            logging.info(f"商品 '{goods_name}' (ID: {gid}) 的 One-Hot 编码正确。")
            print(f"商品 '{goods_name}' (ID: {gid}) 的 One-Hot 编码示例：")
            print(onehot_vector)


def main():
    setup_logging()

    # 定义路径
    goods_id_map_path = '../data/processed/goods_id_map.pkl'
    graph_path = '../data/processed/graphs_with_goods_feature/hetero_graph_2021-11.dgl'  # 替换为实际图文件路径

    # 检查 goods_id_map_path 是否存在
    if not os.path.exists(goods_id_map_path):
        logging.error(f"goods_id_map.pkl 文件不存在: {goods_id_map_path}")
        sys.exit(1)

    # 检查 graph_path 是否存在
    if not os.path.exists(graph_path):
        logging.error(f"图文件不存在: {graph_path}")
        sys.exit(1)

    # 加载 goods_id_map
    goods_id_map = load_goods_id_map(goods_id_map_path)

    # 加载图
    hetero_graph = load_graph(graph_path)

    # 验证 One-Hot 编码
    verify_goods_onehot(hetero_graph, goods_id_map, num_samples=5)


if __name__ == "__main__":
    main()