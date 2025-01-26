import torch
import torch.nn as nn
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

    # 加载新订单数据
    new_orders = pd.read_csv('data/raw/new_orders.csv')
    # 假设 new_orders.csv 已经过清洗和处理，且包含 COMID 和 GOODSNAME
    # 根据你的数据格式调整以下代码

    # 构建异构图（包含新数据）
    # 如果新商户存在，需更新图结构
    # 这里假设所有商户均已存在于图中，简化处理
    # 实际情况可能需要动态更新图

    # 创建商户嵌入（与训练时相同）
    # 加载欺诈标签（假设新数据无标签）

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
        # 假设新订单数据已映射到图中的商户
        # 提取对应商户的特征
        new_comid = new_orders['COMID'].unique()
        new_indices = [idx for idx, comid in enumerate(merchants['COMID']) if comid in new_comid]
        new_embeddings = han_model(hg, hg.nodes['merchant'].data['x'][new_indices])

        # Reprogramming
        new_w_t = reprogram_layer(new_embeddings.unsqueeze(1), W_prime, W_prime).squeeze(1)

        # Attention Fusion
        z_i_t = hg.nodes['merchant'].data['x'][new_indices]

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
        new_u_t = attention_fusion(z_i_t, new_w_t)

        # LLM 增强
        new_llm_output = llm(new_u_t)

        # MLP 预测
        fraud_probabilities = mlp_model(new_llm_output)

    # 将预测结果与商户ID关联
    predictions = fraud_probabilities.cpu().numpy()
    prediction_df = pd.DataFrame({
        'COMID': new_comid,
        'fraud_probability': predictions.flatten()
    })

    # 保存预测结果
    prediction_df.to_csv('data/processed/fraud_predictions.csv', index=False)
    print("欺诈预测结果已保存到 data/processed/fraud_predictions.csv")