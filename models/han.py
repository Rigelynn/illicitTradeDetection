import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv


class HeteroHAN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, metadata, num_heads=8, dropout=0.1):
        super(HeteroHAN, self).__init__()
        self.metadata = metadata

        # 第一层
        self.conv1 = dgl.nn.HeteroGraphConv({
            ('merchant', 'orders', 'tobacco'): GATConv(in_feats, hidden_feats, num_heads=num_heads, feat_drop=dropout,
                                                       attn_drop=dropout, activation=F.elu),
            ('merchant', 'geos', 'merchant'): GATConv(in_feats, hidden_feats, num_heads=num_heads, feat_drop=dropout,
                                                      attn_drop=dropout, activation=F.elu),
            ('merchant', 'similar_orders', 'merchant'): GATConv(in_feats, hidden_feats, num_heads=num_heads,
                                                                feat_drop=dropout, attn_drop=dropout, activation=F.elu),
            ('tobacco', 'rev_orders', 'merchant'): GATConv(in_feats, hidden_feats, num_heads=num_heads,
                                                           feat_drop=dropout, attn_drop=dropout, activation=F.elu),
        }, aggregate='mean')

        # 第二层
        self.conv2 = dgl.nn.HeteroGraphConv({
            ('merchant', 'orders', 'tobacco'): GATConv(hidden_feats * num_heads, out_feats, num_heads=num_heads,
                                                       feat_drop=dropout, attn_drop=dropout, activation=None),
            ('merchant', 'geos', 'merchant'): GATConv(hidden_feats * num_heads, out_feats, num_heads=num_heads,
                                                      feat_drop=dropout, attn_drop=dropout, activation=None),
            ('merchant', 'similar_orders', 'merchant'): GATConv(hidden_feats * num_heads, out_feats,
                                                                num_heads=num_heads, feat_drop=dropout,
                                                                attn_drop=dropout, activation=None),
            ('tobacco', 'rev_orders', 'merchant'): GATConv(hidden_feats * num_heads, out_feats, num_heads=num_heads,
                                                           feat_drop=dropout, attn_drop=dropout, activation=None),
        }, aggregate='mean')

        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, inputs):
        # inputs 是一个字典，包含每种节点类型的特征
        h = self.conv1(graph, inputs)
        h = {k: self.dropout(v) for k, v in h.items()}
        h = self.conv2(graph, h)

        # 对于商户节点，将来自不同边类型的输出拼接
        merchant_feats = torch.cat([
            h[('merchant', 'orders', 'tobacco')],
            h[('merchant', 'geos', 'merchant')],
            h[('merchant', 'similar_orders', 'merchant')],
            h[('tobacco', 'rev_orders', 'merchant')]
        ], dim=1)
        merchant_feats = F.elu(merchant_feats)
        return merchant_feats