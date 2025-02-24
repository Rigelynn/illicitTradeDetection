# src/visualize_graph.py

import dgl
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
from utils import setup_logging


def visualize_graph(hetero_graph, snapshot_str):
    # 将 DGL 异构图转换为 NetworkX 图
    nx_graph = hetero_graph.to_networkx().to_undirected()

    # 定义节点颜色和形状
    color_map = []
    node_size = []
    for node_type in hetero_graph.ntypes:
        for node in hetero_graph.nodes(node_type):
            if node_type == 'merchant':
                color_map.append('skyblue')
                node_size.append(300)
            elif node_type == 'goods':
                color_map.append('lightgreen')
                node_size.append(300)

    # 定义边颜色和样式
    edge_colors = []
    edge_styles = []
    for etype in hetero_graph.etypes:
        if etype == 'ordered':
            edge_colors += ['blue'] * hetero_graph.number_of_edges(etype)
            edge_styles += ['solid'] * hetero_graph.number_of_edges(etype)
        elif etype == 'geographical':
            edge_colors += ['red'] * hetero_graph.number_of_edges(etype)
            edge_styles += ['dashed'] * hetero_graph.number_of_edges(etype)

    pos = nx.spring_layout(nx_graph)

    # 绘制节点
    nx.draw_networkx_nodes(nx_graph, pos, node_color=color_map, node_size=node_size, alpha=0.8)

    # 绘制边
    # 由于 NetworkX 不支持不同类型的边直接用不同样式绘制，我们需要分别绘制
    for etype in hetero_graph.etypes:
        edges = hetero_graph.edges(etype=etype)
        src, dst = edges
        edge_list = zip(src.tolist(), dst.tolist())
        if etype == 'ordered':
            nx.draw_networkx_edges(nx_graph, pos, edgelist=edge_list, edge_color='blue', style='solid', alpha=0.5,
                                   label='Ordered')
        elif etype == 'geographical':
            nx.draw_networkx_edges(nx_graph, pos, edgelist=edge_list, edge_color='red', style='dashed', alpha=0.5,
                                   label='Geographical')

    # 绘制标签
    labels = {}
    labels.update({node: node for node in hetero_graph.nodes('merchant')[:5]})  # 只显示前5个商户标签
    labels.update({node: node for node in hetero_graph.nodes('goods')[:5]})  # 只显示前5个商品标签
    nx.draw_networkx_labels(nx_graph, pos, labels, font_size=8)

    # 显示图例
    plt.legend(scatterpoints=1)
    plt.title(f"Heterogeneous Graph for Snapshot {snapshot_str}")
    plt.axis('off')
    plt.show()


def main():
    setup_logging(log_file='logs/visualize_graph.log')
    import logging
    logging.info("开始可视化异构图。")

    processed_dir = '../data/processed/'
    snapshot_str = '2023-10'  # 替换为你想可视化的时间截面

    graph_path = os.path.join(processed_dir, 'graphs', f'hetero_graph_{snapshot_str}.dgl')
    if not os.path.exists(graph_path):
        logging.error(f"异构图文件未找到: {graph_path}")
        sys.exit(1)

    graphs, _ = dgl.load_graphs(graph_path)
    hetero_graph = graphs[0]

    visualize_graph(hetero_graph, snapshot_str)


if __name__ == "__main__":
    main()