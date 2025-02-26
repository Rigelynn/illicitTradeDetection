from transformers import AutoConfig

# 定义模型路径
model_path = "/home/user4/.llama/checkpoints/Llama3.2-3B"

# 加载模型配置
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

# 打印主要维度参数
print(f"模型名称: {config.architectures}")
print(f"隐藏层大小 (hidden_size): {config.hidden_size}")
print(f"层数 (num_hidden_layers): {config.num_hidden_layers}")
print(f"注意力头数 (num_attention_heads): {config.num_attention_heads}")
print(f"词汇表大小 (vocab_size): {config.vocab_size}")
print(f"最大序列长度 (max_position_embeddings): {config.max_position_embeddings}")
# 如果有其他重要参数，也可以一并打印