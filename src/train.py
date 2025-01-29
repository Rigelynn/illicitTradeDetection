# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.model import FraudDetectionModel
from src.utils import load_preprocessed_data, load_graph, save_model, create_dataloader
from sklearn.model_selection import train_test_split
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
    # 假设每个商户有基础嵌入，可以使用标准化后的订购数量向量
    # 这里示例使用随机向量，实际应根据具体特征计算
    num_merchants = g.num_nodes('merchant')
    merchant_features = torch.randn(num_merchants, configs.han_in_dim).to(device)

    # 加载违规标签
    fraud_labels = df_fraud.sort_values('COMID')['CaseType'].astype(int).values  # 确保与商户节点顺序一致
    y = torch.tensor(fraud_labels, dtype=torch.float32).to(device)

    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(merchant_features, y, test_size=0.2, random_state=42)

    # 创建数据加载器
    batch_size = 64
    train_loader = create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False)

    # 实例化模型
    model = FraudDetectionModel(configs).to(device)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 训练循环
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(g, batch_X, None)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(g, batch_X, None)
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = torch.sigmoid(outputs).round().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
        val_loss /= len(val_loader.dataset)

        # 计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, F1 Score: {f1:.4f}')

    # 保存训练好的模型
    save_path = os.path.join(processed_dir, 'fraud_detection_model.pth')
    save_model(model, save_path)
    print("模型训练完成并已保存。")

if __name__ == "__main__":
    main()