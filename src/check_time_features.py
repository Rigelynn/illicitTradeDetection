import os
import torch


def check_data_consistency(graph_dir):
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pt')])
    if not graph_files:
        raise ValueError("未找到任何 .pt 文件")

    ref_graph = torch.load(os.path.join(graph_dir, graph_files[0]))
    metadata = ref_graph.metadata()
    node_types = metadata[0]
    edge_types = metadata[1]
    feature_dims = {nt: ref_graph[nt].x.size(1) for nt in node_types}

    for file in graph_files[1:]:
        graph = torch.load(os.path.join(graph_dir, file))
        if graph.metadata() != metadata:
            raise ValueError(f"文件 {file} 的元数据不匹配")
        for nt in node_types:
            if graph[nt].x.size(1) != feature_dims[nt]:
                raise ValueError(f"文件 {file} 中节点类型 {nt} 的特征维度不一致")
    print("所有图数据结构一致")


if __name__ == "__main__":
    check_data_consistency("/home/user4/miniconda3/projects/illicitTradeDetection/data/processed/pyg_graphs")