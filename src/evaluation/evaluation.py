#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估模块，计算模型指标：Accuracy、Precision、Recall、F1、AUC 以及混淆矩阵
"""
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            for sample in batch:
                merchant_id, sequence, aggregated_prompts, label = sample
                sequence = sequence.to(device)
                label = label.to(device)
                pred = model(sequence, aggregated_prompts)
                prob = torch.sigmoid(pred).item()
                pred_bin = 1 if prob >= 0.5 else 0
                all_preds.append(pred_bin)
                all_labels.append(int(label.item()))
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, precision, recall, f1, auc, cm