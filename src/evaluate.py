# src/evaluate.py

import torch
from src.model import FraudDetectionModel
from src.utils import load_graph, load_preprocessed_data, load_model
from sklearn.metrics import classification_report
import os
import pandas as pd

def main():
    # 配置参数
    class Configs:
        # 任务相关
        task_name = 'fraud_detection'
        pred_len = 10
        seq_len = 50
        enc_in = 128  # 输入特征维度（商户嵌入维度）

        # HAN相关
        han_in_dim = 128
        han_hidden_dim = 256
        han_num_layers = 2

        # Reprogramming相关
        reprog_n_heads = 8
        reprog_d_tokens = 1000

        # LLM相关
        llm_model = 'LLAMA'  # 可选 'LLAMA', 'GPT2', 'BERT'
        llm_dim = 768
        llm_layers = 12

        # Patch Embedding
        patch_len = 16
        stride = 8
        d_model = 256
        dropout = 0.1

        # 输出相关
        output_dim = 1

    configs = Configs()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 加载预处理数据和异构图
    processed_dir = '../data/processed/'
    df_ordering, df_fraud = load_preprocessed_data(processed_dir)
    g = load_graph(processed_dir).to(device)

    # 初始化节点特征
    num_merchants = g.num_nodes('merchant')
    merchant_features = torch.randn(num_merchants, configs.han_in_dim).to(device)

    # 加载违规标签
    fraud_labels = df_fraud.sort_values('COMID')['CaseType'].astype(int).values  # 确保与商户节点顺序一致
    y = torch.tensor(fraud_labels, dtype=torch.float32).to(device)

    # 实例化模型
    model = FraudDetectionModel(configs).to(device)

    # 加载训练好的模型
    model_path = os.path.join(processed_dir, 'fraud_detection_model.pth')
    load_model(model, model_path, device)

    # 直接使用全部数据作为测试集（根据需求可调整）
    model.eval()
    with torch.no_grad():
        outputs = model(g, merchant_features, None)
        outputs = torch.sigmoid(outputs).squeeze(-1)
        preds = (outputs > 0.5).cpu().numpy()
        all_preds = preds
        all_labels = y.cpu().numpy()

    # 评估
    report = classification_report(all_labels, all_preds, digits=4)
    print("分类报告：")
    print(report)

if __name__ == "__main__":
    main()