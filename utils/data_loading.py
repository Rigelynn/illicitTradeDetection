# utils/data_loading.py

import torch
import pandas as pd
import dgl


def load_graph_and_mappings(graph_path, merchants_path, tobaccos_path):
    """
    加载异构图以及商户和烟草类型的映射文件。

    参数：
        graph_path (str): 异构图文件的路径（.pt文件）。
        merchants_path (str): 商户映射文件的路径（.csv文件）。
        tobaccos_path (str): 烟草类型映射文件的路径（.csv文件）。

    返回：
        hg (dgl.DGLHeteroGraph): 加载的异构图。
        merchants (pd.DataFrame): 商户信息。
        tobaccos (pd.Index): 烟草类型信息。
    """
    # 加载异构图
    hg = torch.load(graph_path)
    print(f"异构图加载完成，包含节点类型: {hg.ntypes}，边类型: {hg.etypes}")

    # 加载商户和烟草类型映射
    merchants = pd.read_csv(merchants_path)
    tobaccos = pd.read_csv(tobaccos_path, header=None).squeeze()  # 读取为 Series
    print(f"商户数量: {len(merchants)}，烟草类型数量: {len(tobaccos)}")

    return hg, merchants, tobaccos


def load_fraud_labels(fraud_labels_path):
    """
    加载欺诈标签数据。

    参数：
        fraud_labels_path (str): 欺诈标签文件的路径（.xlsx文件）。

    返回：
        fraud_labels_df (pd.DataFrame): 欺诈标签数据。
    """
    fraud_labels_df = pd.read_excel(fraud_labels_path)
    print(f"欺诈标签数据加载完成，记录数: {len(fraud_labels_df)}")
    return fraud_labels_df