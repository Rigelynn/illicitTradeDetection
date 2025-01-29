# main.py

from utils.model_loading import Model
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Configs:
    def __init__(self):
        self.task_name = "TestTask"
        self.pred_len = 100
        self.seq_len = 512
        self.d_ff = 2048
        self.llm_dim = 4096

def main():
    # 创建配置对象
    configs = Configs()

    # 初始化模型
    model = Model(configs)

    # 示例提示词
    prompt = "这是一个测试句子。"

    # 生成文本
    generated_text = model.generate_text(prompt, max_length=50)
    print("生成的文本：", generated_text)

if __name__ == "__main__":
    main()