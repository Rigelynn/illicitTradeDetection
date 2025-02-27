#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer


def convert_official_to_hf(official_model_dir, output_dir):
    # 官方模型权重文件路径（假设权重合并成一个文件：consolidated.00.pth）
    checkpoint_path = os.path.join(official_model_dir, "consolidated.00.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到权重文件: {checkpoint_path}")

    print("加载官方权重文件...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 如果需要对 state_dict 键名进行转换，请在此处实现，例如：
    # state_dict = rename_state_dict_keys(state_dict)
    # 此处默认官方权重键名和 HuggingFace 模型能够直接对应（或经过小幅适配）

    print("定义模型配置...")
    # 根据实际情况确定模型的配置参数，下例参数为 LLaMA-2-7B 模型的典型配置
    config_dict = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "use_cache": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0
    }
    config = AutoConfig.from_dict(config_dict)

    print("初始化空模型...")
    model = AutoModelForCausalLM.from_config(config)

    print("加载权重到模型...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print("缺失的键:", missing_keys)
    if unexpected_keys:
        print("多余的键:", unexpected_keys)
    print("权重加载完成！")

    # 保存为 Hugging Face 标准模型格式
    print("保存转换后的模型到:", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    # 如果官方模型目录中有分词器文件 tokenizer.model，则复制并生成对应的 tokenizer 配置
    tokenizer_file = os.path.join(official_model_dir, "tokenizer.model")
    if os.path.exists(tokenizer_file):
        # 这里直接使用 LlamaTokenizer 加载，并保存到模型目录中
        try:
            tokenizer = LlamaTokenizer.from_pretrained(official_model_dir, legacy=False)
        except Exception as e:
            print("加载官方 tokenizer 出错，尝试直接复制 tokenizer 文件...", e)
            import shutil
            shutil.copy(tokenizer_file, os.path.join(output_dir, "tokenizer.model"))
        else:
            tokenizer.save_pretrained(output_dir)

    print("模型转换完成！")


if __name__ == "__main__":
    # 设置官方模型的目录和转换后模型保存路径
    official_model_directory = "/mnt/hdd/huggingface/Llama-2-7b"  # 官方模型文件所在目录
    output_directory = "/mnt/hdd/huggingface/llama2_hf"  # 保存转换后 Hugging Face 格式模型的目录
    convert_official_to_hf(official_model_directory, output_directory)