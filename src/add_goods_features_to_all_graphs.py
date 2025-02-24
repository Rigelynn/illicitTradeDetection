import os
import sys
import logging
import pickle
import torch
import dgl


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


def add_goods_onehot_features_to_graph(hetero_graph, goods_id_map):
    """
    为异构图中的 'goods' 节点添加 One-Hot 编码特征。

    参数：
        hetero_graph: dgl 异构图
        goods_id_map: dict，商品名称到唯一整数 ID 的映射
    """
    num_goods = len(goods_id_map)

    # 确保按商品 ID 排序
    sorted_goods = sorted(goods_id_map.items(), key=lambda x: x[1])

    # 生成 One-Hot 编码矩阵
    onehot_matrix = torch.eye(num_goods, dtype=torch.float32)

    # 添加到图中
    hetero_graph.nodes['goods'].data['goods_onehot'] = onehot_matrix
    logging.info(f"为 'goods' 节点添加了 One-Hot 编码特征，形状：{onehot_matrix.shape}")


def process_graph_file(graph_path, goods_id_map, output_dir):
    """
    处理单个图文件：加载、添加特征、保存。

    参数：
        graph_path: str，原始图文件路径
        goods_id_map: dict，商品名称到唯一整数 ID 的映射
        output_dir: str，更新后图文件的保存目录
    """
    try:
        # 加载图
        graphs, _ = dgl.load_graphs(graph_path)
        hetero_graph = graphs[0]
        logging.info(f"加载图文件：{graph_path}，包含节点类型：{hetero_graph.ntypes}，边类型：{hetero_graph.etypes}")

        # 添加商品特征
        add_goods_onehot_features_to_graph(hetero_graph, goods_id_map)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 构建输出文件路径
        graph_filename = os.path.basename(graph_path)
        output_path = os.path.join(output_dir, graph_filename)

        # 保存更新后的图
        dgl.save_graphs(output_path, [hetero_graph])
        logging.info(f"保存更新后的图到：{output_path}")

    except Exception as e:
        logging.error(f"处理图文件 {graph_path} 时出错：{e}")


def main():
    setup_logging()

    # 定义路径
    graphs_dir = '../data/processed/graphs'  # 更新后的路径
    goods_id_map_path = '../data/processed/goods_id_map.pkl'
    output_dir = '../data/processed/graphs_with_goods_feature'

    # 打印当前工作目录（用于调试）
    current_dir = os.getcwd()
    logging.info(f"当前工作目录: {current_dir}")

    # 检查 graphs_dir 是否存在
    if not os.path.exists(graphs_dir):
        logging.error(f"图目录不存在: {graphs_dir}")
        sys.exit(1)

    # 检查 goods_id_map_path 是否存在
    if not os.path.exists(goods_id_map_path):
        logging.error(f"goods_id_map.pkl 文件不存在: {goods_id_map_path}")
        sys.exit(1)

    # 加载 goods_id_map
    goods_id_map = load_goods_id_map(goods_id_map_path)

    # 获取所有 .dgl 文件
    graph_files = [f for f in os.listdir(graphs_dir) if f.endswith('.dgl')]
    logging.info(f"找到 {len(graph_files)} 个图文件需要处理。")

    # 遍历并处理每个图文件
    for graph_file in graph_files:
        graph_path = os.path.join(graphs_dir, graph_file)
        logging.info(f"开始处理图文件：{graph_path}")
        process_graph_file(graph_path, goods_id_map, output_dir)

    logging.info("所有图文件处理完成。")


if __name__ == "__main__":
    main()