import torch
import torch.nn as nn
import torch.optim as optim
import dgl
from tqdm import tqdm
import pandas as pd
import numpy as np

from models.han import HeteroHAN
from models.reprogramming import ReprogrammingLayer
from models.mlp import MLP
from utils.graph_construction import construct_heterogeneous_graph
from utils.data_loading import load_graph_and_mappings
from utils.helpers import load_prototypes


def load_prototypes(file_path):
    W_prime = np.load(file_path)
    W_prime = torch.tensor(W_prime, dtype=torch.float)
    return W_prime


def main():
    # 配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {
        'graph_path': 'data/processed/hetero_graph.pt',
        'prototypes_path': 'data/prototypes/W_prime.npy',
        'merchants_path': 'data/processed/merchants.csv',
        'tobaccos_path': 'data/processed/tobaccos.csv',
        'epochs': 50,
        'learning_rate': 0.001,
        'batch_size': 256,
        'epsilon': 0.8
    }

    # 加载图
    print("加载异构图...")
    hg = torch.load(config['graph_path'])
    hg = hg.to(device)

    # 加载商户和烟草类型映射
    merchants = pd.read_csv(config['merchants_path'])
    tobaccos = pd.read_csv(config['tobaccos_path'])

    # 假设欺诈标签在 'data/raw/fraud_labels.xlsx'
    fraud_labels_df = pd.read_excel('data/raw/fraud_labels.xlsx')

    # 合并欺诈标签与商户信息
    merged_df = merchants.merge(fraud_labels_df, on='COMID', how='left')
    merged_df['fraud_label'] = merged_df['CaseType'].notna().astype(int)

    # 数据拆分为训练集、验证集和测试集（70%、15%、15%）
    train_df = merged_df.sample(frac=0.7, random_state=42)
    temp_df = merged_df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    # 创建标签张量
    train_labels = torch.tensor(train_df['fraud_label'].values, dtype=torch.float).to(device)
    val_labels = torch.tensor(val_df['fraud_label'].values, dtype=torch.float).to(device)
    test_labels = torch.tensor(test_df['fraud_label'].values, dtype=torch.float).to(device)

    # 加载并准备原型
    W_prime = load_prototypes(config['prototypes_path']).to(device)

    # 定义模型组件
    in_feats = hg.nodes['merchant'].data['x'].shape[1]  # 节点特征维度
    hidden_feats = 128
    out_feats = 256
    metadata = hg.metadata()

    han_model = HeteroHAN(in_feats, hidden_feats, out_feats, metadata, num_heads=8, dropout=0.1).to(device)

    reprogram_layer = ReprogrammingLayer(d_model=out_feats, n_heads=8, d_keys=16, d_llm=768).to(device)

    mlp_model = MLP(input_dim=768, hidden_dim=256).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(
        list(han_model.parameters()) + list(reprogram_layer.parameters()) + list(mlp_model.parameters()),
        lr=config['learning_rate'])
    criterion = nn.BCELoss()

    # 训练循环
    for epoch in range(1, config['epochs'] + 1):
        han_model.train()
        reprogram_layer.train()
        mlp_model.train()

        # 前向传播通过 HAN
        merchant_embeddings = han_model(hg, hg.nodes['merchant'].data['x'])

        # Reprogramming
        w_t = reprogram_layer(merchant_embeddings.unsqueeze(1), W_prime, W_prime)  # (num_merchants, 1, 768)
        w_t = w_t.squeeze(1)  # (num_merchants, 768)

        # Attention Fusion
        z_i_t = hg.nodes['merchant'].data['x']  # 基础信息嵌入

        class AttentionFusion(nn.Module):
            def __init__(self, embed_dim):
                super(AttentionFusion, self).__init__()
                self.query = nn.Linear(embed_dim, embed_dim)
                self.key = nn.Linear(embed_dim, embed_dim)
                self.value = nn.Linear(embed_dim, embed_dim)
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, z, w):
                Q = self.query(z)  # (B, embed_dim)
                K = self.key(w)  # (V', embed_dim)
                V = self.value(w)  # (V', embed_dim)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(Q.size(-1))  # (B, V')
                attn = self.softmax(scores)  # (B, V')
                out = torch.matmul(attn, V)  # (B, embed_dim)
                return out

        attention_fusion = AttentionFusion(embed_dim=768).to(device)
        u_t = attention_fusion(z_i_t, w_t)  # (num_merchants, 768)

        # 创建 patch（此处简化为每个样本一个 patch）
        patches = u_t.unsqueeze(1)  # (num_merchants, 1, 768)

        # 构建 prompt（这里暂时简化，不使用真实的文本 prompt）
        prompts = []
        for patch in patches:
            patch_features = ','.join(map(str, patch.squeeze(1).tolist()))
            prompt = f"Analyze the following transaction patterns: {patch_features}"
            prompts.append(prompt)

        # 加载 LLM 并进行处理（这里使用一个简单的模拟 LLM）
        class DummyLLM(nn.Module):
            def __init__(self, d_llm=768):
                super(DummyLLM, self).__init__()
                self.linear = nn.Linear(768, 768)

            def forward(self, x):
                return torch.relu(self.linear(x))

        llm = DummyLLM().to(device)
        llm.eval()
        with torch.no_grad():
            llm_output = llm(u_t)  # (num_merchants, 768)

        # MLP 预测
        predictions = mlp_model(llm_output)  # (num_merchants, 1)

        # 计算损失
        loss = criterion(predictions, train_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每隔一定周期进行验证
        if epoch % 5 == 0 or epoch == 1:
            han_model.eval()
            reprogram_layer.eval()
            mlp_model.eval()

            with torch.no_grad():
                # 前向传播验证集
                val_embeddings = han_model(hg, hg.nodes['merchant'].data['x'][val_df.index])
                val_w_t = reprogram_layer(val_embeddings.unsqueeze(1), W_prime, W_prime).squeeze(1)
                val_z_i_t = hg.nodes['merchant'].data['x'][val_df.index]
                val_u_t = attention_fusion(val_z_i_t, val_w_t)
                val_llm_output = llm(val_u_t)
                val_predictions = mlp_model(val_llm_output)

                # 计算验证损失
                val_loss = criterion(val_predictions, val_labels)

                # 计算验证指标
                pred_labels = (val_predictions >= 0.5).float()
                accuracy = (pred_labels == val_labels).float().mean().item()
                precision = ((pred_labels * val_labels) / (pred_labels.sum() + 1e-8)).mean().item()
                recall = ((pred_labels * val_labels) / (val_labels.sum() + 1e-8)).mean().item()

                print(
                    f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, Val Acc = {accuracy:.4f}, Val Precision = {precision:.4f}, Val Recall = {recall:.4f}")

        # 保存模型（可以根据需求调整保存频率）
        torch.save(han_model.state_dict(), "models/han.pth")
        torch.save(reprogram_layer.state_dict(), "models/reprogramming.pth")
        torch.save(mlp_model.state_dict(), "models/mlp.pth")


if __name__ == "__main__":
    main()