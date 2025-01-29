# utils/model_loading.py

import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(model_name: str):
    """
    使用 transformers 库从 Hugging Face Hub 加载模型和分词器。

    参数：
        model_name (str): 模型在 Hugging Face Hub 上的标识符，例如 "huggyllama/llama-7b"

    返回：
        tokenizer: 加载的分词器。
        model: 加载的语言模型。
        device: 模型所在设备。
    """
    try:
        logging.info(f"开始加载模型和分词器: {model_name}")

        # 确定使用的设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"使用的设备: {device}")

        # 加载分词器
        logging.info("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logging.info("分词器加载成功。")

        # 加载模型
        logging.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto' if device == 'cuda' else None,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32
        )
        logging.info("模型加载成功。")

        # 将模型移动到指定设备（如果未使用 `device_map='auto'`）
        if device == 'cpu':
            model.to(device)
        model.eval()

        logging.info("模型和分词器加载完成。")
        return tokenizer, model, device

    except Exception as e:
        logging.error(f"加载模型时出错: {e}")
        raise e

class Model:
    def __init__(self, model_name: str, patch_len=16, stride=8):
        """
        初始化 Model 类，加载指定的语言模型和分词器。

        参数：
            model_name (str): 模型在 Hugging Face Hub 上的标识符。
            patch_len (int): 补丁长度（根据需求）。
            stride (int): 步幅（根据需求）。
        """
        self.patch_len = patch_len
        self.stride = stride

        # 加载模型和分词器
        self.tokenizer, self.llm_model, self.device = load_model(model_name)

    def generate_text(self, prompt: str, max_length: int = 50):
        """
        使用语言模型生成文本。

        参数：
            prompt (str): 输入提示词。
            max_length (int): 生成文本的最大长度。

        返回：
            str: 生成的文本。
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            logging.error(f"生成文本时出错: {e}")
            return ""