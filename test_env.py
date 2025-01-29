# test_env.py
import torch
import pandas as pd
import dgl
from transformers import BertTokenizer, BertModel



def test():
    print("PyTorch版本:", torch.__version__)
    print("CUDA是否可用:", torch.cuda.is_available())
    print("CUDA版本:", torch.version.cuda)
    print("当前GPU设备:", torch.cuda.current_device())
    print("GPU名称:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # 测试 Transformers
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model(**inputs)
    print("Transformers加载并工作正常。")

    # 测试 DGL
    g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
    print("DGL图:", g)

    # 测试 pandas
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    print("pandas DataFrame:\n", df)




if __name__ == "__main__":
    test()