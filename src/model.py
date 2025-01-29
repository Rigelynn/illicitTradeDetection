# src/model.py

import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaTokenizer, GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
from math import sqrt
from dgl.nn.pytorch import HeteroGraphConv
from dgl.nn.pytorch import GraphConv


class HANLayer(nn.Module):
    """
    HAN (Heterogeneous Attention Network) Layer
    处理基于元路径的卷积和聚合。
    """

    def __init__(self, in_dim, out_dim, meta_paths):
        super(HANLayer, self).__init__()
        self.meta_paths = meta_paths
        self.conv = nn.ModuleDict({
            'ordering': nn.Linear(in_dim, out_dim),
            'geographical': nn.Linear(in_dim, out_dim)
        })
        self.attn = nn.ModuleDict({
            mp: nn.Linear(out_dim, 1) for mp in meta_paths
        })

    def forward(self, g, h):
        path_convs = {}
        for mp in self.meta_paths:
            # 基于元路径提取子图
            try:
                path_g = g.metapath(mp)
            except KeyError:
                print(f"元路径 {mp} 在图中未找到，跳过。")
                continue

            # 提取相关节点特征
            h_path = h[mp[0]]  # 假设所有节点类型具有相同的特征
            # 简单地使用线性层模拟GraphConv
            conv = self.conv[mp[1]](h_path)
            path_convs[mp] = conv

        # 聚合不同元路径的卷积结果
        h_new = {}
        for etype in g.etypes:
            if etype in path_convs:
                h_new[etype] = path_convs[etype]

        return h_new


class HAN(nn.Module):
    """
    HAN (Heterogeneous Attention Network) 模型
    堆叠多个HAN层并聚合元路径信息。
    """

    def __init__(self, in_dim, hidden_dim, meta_paths, num_layers=2):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HANLayer(in_dim, hidden_dim, meta_paths))
            in_dim = hidden_dim  # 更新下一层的输入维度

        # 不同元路径的注意力权重
        self.attention = nn.Parameter(torch.Tensor(len(meta_paths), hidden_dim))
        nn.init.xavier_uniform_(self.attention)

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)

        # 聚合不同元路径的嵌入表示
        h_combined = {}
        for ntype in h:
            # 叠加所有元路径的嵌入（简单平均）
            h_combined[ntype] = torch.mean(torch.stack([h[ntype] for _ in range(len(self.attention))]), dim=0)
        return h_combined


class ReprogrammingLayer(nn.Module):
    """
    Reprogramming Layer，用于将图嵌入转换为LLM可处理的词嵌入。
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


class FraudDetectionModel(nn.Module):
    """
    结合HAN和LLM的欺诈检测模型。
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super(FraudDetectionModel, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # 初始化HAN
        self.meta_paths = [
            ('merchant', 'orders', 'goods'),
            ('merchant', 'geographical', 'merchant')
        ]
        self.han = HAN(in_dim=configs.han_in_dim, hidden_dim=configs.han_hidden_dim, meta_paths=self.meta_paths,
                       num_layers=configs.han_num_layers)

        # 初始化LLM
        self.llm_model, self.tokenizer = self.init_llm(configs)

        # 设置pad token
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 冻结LLM参数
        for param in self.llm_model.parameters():
            param.requires_grad = False

        # ReprogrammingLayer 初始化
        self.reprogramming_layer = ReprogrammingLayer(configs.han_hidden_dim, configs.reprog_n_heads, d_llm=self.d_llm)

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)

        # Mapping Layer
        self.mapping_layer = nn.Linear(self.llm_model.config.hidden_size, configs.reprog_d_tokens)

        # 输出层
        self.output_projection = nn.Sequential(
            nn.Linear(configs.reprog_d_tokens, configs.output_dim),
            nn.Dropout(configs.dropout)
        )

        # 标准化层
        self.normalize = Normalize(configs.han_hidden_dim)

    def init_llm(self, configs):
        """
        根据配置初始化LLM模型和分词器。
        """
        if configs.llm_model == 'LLAMA':
            llm = LlamaModel.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False,
            )
            tokenizer = LlamaTokenizer.from_pretrained(
                'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=False
            )
        elif configs.llm_model == 'GPT2':
            llm = GPT2Model.from_pretrained(
                'gpt2',
                trust_remote_code=True,
                local_files_only=False,
            )
            tokenizer = GPT2Tokenizer.from_pretrained(
                'gpt2',
                trust_remote_code=True,
                local_files_only=False
            )
        elif configs.llm_model == 'BERT':
            llm = BertModel.from_pretrained(
                'bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
            )
            tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False
            )
        else:
            raise ValueError("Unsupported LLM model specified.")
        return llm, tokenizer

    def forward(self, g, node_features, node_types):
        """
        前向传播，结合HAN和LLM进行欺诈检测。
        """
        # 通过HAN获取商户嵌入
        h = self.han(g, node_features)
        merchant_h = h['merchant']  # 假设 'merchant' 是节点类型

        # Reprogramming: 将商户嵌入转换为LLM词嵌入
        w_i = self.reprogramming_layer(merchant_h, self.llm_model.get_input_embeddings().weight,
                                       self.llm_model.get_input_embeddings().weight)

        # 准备时间序列序列
        # 假设 node_features 已经按时间排序，并且每个商户的序列被正确组织
        # 此处需要根据实际数据格式进行调整

        # Tokenization: 将嵌入转换为序列 token
        # 这里简单示例，将连续的嵌入作为序列输入到LLM
        # 实际上可能需要更复杂的处理，如生成文本 prompts
        prompt = "Fraud detection prompt"  # 示例 prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(merchant_h.device)
        llm_outputs = self.llm_model(**inputs).last_hidden_state

        # Patch Embedding
        patches, patch_lengths = self.patch_embedding(llm_outputs)

        # 通过输出投影得到预测
        predictions = self.output_projection(patches)

        return predictions