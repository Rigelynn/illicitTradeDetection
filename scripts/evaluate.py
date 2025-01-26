import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

from models.han import HeteroHAN
from models.reprogramming import ReprogrammingLayer
from models.mlp import MLP
from utils.graph_construction import construct_heterogeneous_graph
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
    }

    # 加载图
    print("加载异构图...")
    hg = torch.load(config['graph_path'])
    hg = hg.to(device)

    # 加载商户和烟草类型映射
    merchants = pd.read_csv(config['merchants_path'])
    tobaccos = pd.read_csv(config['tobaccos_path'])

    # 加载测试标签
    fraud_labels_df = pd.read_excel('data/raw/fraud_labels.xlsx')
    merged_df = merchants.merge(fraud_labels_df, on='COMID', how='left')
    merged_df['fraud_label'] = merged_df['CaseType'].notna().astype(int)

    # 数据拆分为训练集、验证集和测试集（70%、15%、15%）
    train_df = merged_df.sample(frac=0.7, random_state=42)
    temp_df = merged_df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)

    # 创建标签张量
    test_labels = torch.tensor(test_df['fraud_label'].values, dtype=torch.float).to(device)

    # 加载并准备原型
    W_prime = load_prototypes(config['prototypes_path']).to(device)

    # 定义模型组件
    in_feats = hg.nodes['merchant'].data['x'].shape[1]
    hidden_feats = 128
    out_feats = 256
    metadata = hg.metadata()

    han_model = HeteroHAN(in_feats, hidden_feats, out_feats, metadata, num_heads=8, dropout=0.1).to(device)
    han_model.load_state_dict(torch.load('models/han.pth'))
    han_model.eval()

    reprogram_layer = ReprogrammingLayer(d_model=out_feats, n_heads=8, d_keys=16, d_llm=768).to(device)
    reprogram_layer.load_state_dict(torch.load('models/reprogramming.pth'))
    reprogram_layer.eval()

    mlp_model = MLP(input_dim=768, hidden_dim=256).to(device)
    mlp_model.load_state_dict(torch.load('models/mlp.pth'))
    mlp_model.eval()

    # 加载 LLM（模拟）
    class DummyLLM(nn.Module):
        def __init__(self, d_llm=768):
            super(DummyLLM, self).__init__()
            self.linear = nn.Linear(768, 768)

        def forward(self, x):
            return torch.relu(self.linear(x))

    llm = DummyLLM().to(device)
    llm.eval()

    # 前向传播
    with torch.no_grad():
        # HAN 嵌入
        test_embeddings = han_model(hg, hg.nodes['merchant'].data['x'][test_df.index])

        # Reprogramming
        test_w_t = reprogram_layer(test_embeddings.unsqueeze(1), W_prime, W_prime).squeeze(1)

        # Attention Fusion
        z_i_t = hg.nodes['merchant'].data['x'][test_df.index]

        class AttentionFusion(nn.Module):
            def __init__(self, embed_dim):
                super(AttentionFusion, self).__init__()
                self.query = nn.Linear(embed_dim, embed_dim)
                self.key = nn.Linear(embed_dim, embed_dim)
                self.value = nn.Linear(embed_dim, embed_dim)
                self.softmax = nn.Softmax(dim=-1)

            def forward(self, z, w):
                Q = self.query(z)
                K = self.key(w)
                V = self.value(w)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(Q.size(-1))
                attn = self.softmax(scores)
                out = torch.matmul(attn, V)
                return out

        attention_fusion = AttentionFusion(embed_dim=768).to(device)
        test_u_t = attention_fusion(z_i_t, test_w_t)

        # LLM 增强
        llm_output = llm(test_u_t)

        # MLP 预测
        test_predictions = mlp_model(llm_output)

    # 将预测结果和真实标签转换到 CPU
    preds = test_predictions.cpu().numpy()
    labels = test_labels.cpu().numpy()
    binary_preds = (preds >= 0.5).astype(int)

    # 计算评估指标
    accuracy = accuracy_score(labels, binary_preds)
    precision = precision_score(labels, binary_preds)
    recall = recall_score(labels, binary_preds)
    f1 = f1_score(labels, binary_preds)
    auc = roc_auc_score(labels, preds)

    print(f"测试准确率: {accuracy:.4f}")
    print(f"测试精确率: {precision:.4f}")
    print(f"测试召回率: {recall:.4f}")
    print(f"测试F1分数: {f1:.4f}")
    print(f"测试AUC-ROC: {auc:.4f}")