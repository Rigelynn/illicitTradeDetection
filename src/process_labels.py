#!/usr/bin/env python
# process_labels.py

import pandas as pd


def process_label_data(file_path):
    """
    处理原始的标签 Excel 数据。
    1. 读取 Excel。
    2. 去除可能重复的表头行。
    3. 根据 LICEN_NO 判断：有值的认为是违规的（label=1），否则为非违规（label=0）。
    4. 对违规的商户进行汇总，例如将多条违规记录（CASE_REASON, CASE_VALUE, CASE_TIME）整合成列表。

    返回：
        df_raw: 原始的 DataFrame（包含全部记录）
        merchant_label_dict: 以商户编号（LICEN_NO）为 key 的违规信息字典
    """
    # 读取 Excel（注意：Excel 文件中若存在多余的表头需要处理）
    df = pd.read_excel(file_path)
    # 去除重复的表头行（例如行中 LICEN_NO 为 'LICEN_NO' 的记录），确保所有数据行都有正确字段
    df = df[df['LICEN_NO'] != 'LICEN_NO']
    # 判断违规：LICEN_NO 非空则视为违规
    df['label'] = df['LICEN_NO'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0)

    # 对违规的商户进行汇总
    df_violation = df[df['label'] == 1].groupby('LICEN_NO', as_index=False).agg({
        'CASE_REASON': lambda reasons: list(reasons),
        'CASE_VALUE': 'sum',  # 可使用 sum 或其它聚合方式（如 mean）
        'CASE_TIME': lambda times: list(times)
    })
    merchant_label_dict = df_violation.set_index('LICEN_NO').to_dict(orient='index')
    return df, merchant_label_dict


if __name__ == '__main__':
    file_path = "../data/raw/illicitInfo.xlsx"
    df, merchant_label_dict = process_label_data(file_path)
    print("原始数据示例：")
    print(df.head())
    print("汇总后的违规标签：")
    print(merchant_label_dict)