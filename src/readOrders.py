import torch

# 加载 integrated_embeddings 文件
embeddings_path = "/mnt/hdd/user4data/integrated_embeddings/integrated_embeddings_2021-11.pth"
integrated_embeddings = torch.load(embeddings_path)

# 查看文件中存储的所有商户ID
print("所有商户ID:", integrated_embeddings.keys())

# 查看其中一个商户的嵌入向量维度
for merchant_id, embedding in integrated_embeddings.items():
    print(f"商户 {merchant_id} 的嵌入维度: {embedding.shape}")
    break  # 只查看一个样例