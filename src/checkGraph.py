#!/usr/bin/env python
# check_hetero_graph.py

import os
import torch


def check_hetero_graph(graph_path):
    if not os.path.exists(graph_path):
        print(f"文件不存在：{graph_path}")
        return

    try:
        graph = torch.load(graph_path, map_location="cpu")
    except Exception as e:
        print(f"加载图文件出错: {e}")
        return

    print("成功加载图文件！")
    print("图对象类型:", type(graph))

    # 检查是否为 HeteroData 对象（来自 torch_geometric.data.HeteroData）
    # 如果不是 HeteroData，也可能是字典形式保存的数据，依情况调整检查逻辑
    if hasattr(graph, 'node_types'):
        print("图中的节点类型:")
        print(graph.node_types)
    else:
        print("该图对象没有 node_types 属性，可能不是标准的 HeteroData 对象。")
        print("图的 keys:", list(graph.keys()))

    # 尝试调用 metadata() 方法——元信息一般包括 (node_types, edge_types)
    if hasattr(graph, "metadata"):
        try:
            metadata = graph.metadata()
            print("图的元信息 (node_types, edge_types):")
            print(metadata)
        except Exception as e:
            print("获取图元信息时出错:", e)
    else:
        print("图对象没有 metadata() 方法。")

    # 针对每个节点类型，检查其存储的内容
    node_types = None
    if hasattr(graph, 'node_types'):
        node_types = graph.node_types
    else:
        # 如果 graph 是 dict 格式，则认为顶层的 key 为节点类型之一
        node_types = list(graph.keys())

    for nt in node_types:
        try:
            node_data = graph[nt]
            print(f"\n节点类型: {nt}")
            # 如果该节点数据是一个对象（常用 HeteroData 中为 Data），检查其中的属性
            if hasattr(node_data, "keys"):
                print("属性键值:", list(node_data.keys()))

                # 如果有特征键 "x"，打印特征维度信息
                if "x" in node_data:
                    x = node_data.x
                    print(f"特征'x'形状: {x.shape}")
                else:
                    print("没有找到特征键 'x'")
            # 如果是其它类型，则直接打印内容
            else:
                print("该节点数据为:", node_data)
        except Exception as e:
            print(f"访问节点类型 {nt} 时出错：", e)


if __name__ == "__main__":
    # 指定图文件路径（相对路径或绝对路径）
    graph_file_path = "../data/processed/pyg_graphs/hetero_graph_2021-11.pt"
    check_hetero_graph(graph_file_path)