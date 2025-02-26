#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast

from transformers import AutoConfig

config = AutoConfig.from_pretrained("/mnt/hdd/huggingface/qwen/Qwen-14B-Chat")
print("hidden_size:", config.hidden_size)

def check_modeling_qwen(path):
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        tree = ast.parse(content, filename=path)
    except Exception as e:
        print("解析文件时出错：", e)
        return

    class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    print("以下是文件中定义的所有类：")
    for name in class_names:
        print(" -", name)


if __name__ == '__main__':
    file_path = "/mnt/hdd/huggingface/qwen/Qwen-14B-Chat/modeling_qwen.py"
    check_modeling_qwen(file_path)

