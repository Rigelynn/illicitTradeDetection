# process_fraud_labels_time.py

import os
import pandas as pd
import pickle


def load_fraud_labels(fraud_labels_path):
    """
    加载违规标签数据，并解析 CASE_TIME 列为 datetime 类型。
    """
    df_fraud = pd.read_csv(fraud_labels_path, parse_dates=['CASE_TIME'])
    return df_fraud


def load_merchant_id_map(merchant_id_map_path):
    """
    加载商户 ID 映射，该映射为 {LICEN_NO: merchant_id} 的字典。
    """
    with open(merchant_id_map_path, 'rb') as f:
        merchant_id_map = pickle.load(f)
    return merchant_id_map


def construct_labels_for_snapshot(df_fraud, merchant_id_map, snapshot_str):
    """
    针对给定的时间截面（例如 "2021-11"）构造每个商户的标签。

    对于每个商户，
      - 如果其 LICEN_NO 在该时间截面内至少出现一次违规记录，则标记为正例 (1)；
      - 否则标记为反例 (0)。

    参数：
      df_fraud      : pd.DataFrame，违规标签数据，要求 CASE_TIME 为 datetime 类型；
      merchant_id_map: dict，{LICEN_NO: merchant_id}；
      snapshot_str  : str，时间截面，格式为 "YYYY-MM"（例如 "2021-11"）。

    返回：
      pd.DataFrame，包含 'merchant_id', 'LICEN_NO' 和 'label' 三列。
    """
    # 过滤出该时间截面内的违规记录
    df_snapshot = df_fraud[df_fraud['CASE_TIME'].dt.strftime('%Y-%m') == snapshot_str]
    fraud_licen_no = df_snapshot['LICEN_NO'].unique()

    labels = []
    for licen_no, merchant_id in merchant_id_map.items():
        label = 1 if licen_no in fraud_licen_no else 0
        labels.append({
            'merchant_id': merchant_id,
            'LICEN_NO': licen_no,
            'label': label
        })

    df_labels = pd.DataFrame(labels)
    return df_labels


def main():
    # 定义违规标签和商户映射的路径
    fraud_labels_path = '../data/processed/fraud_labels_processed.csv'
    merchant_id_map_path = '../data/processed/merchant_id_map.pkl'

    # 输出目录（按时间截面保存标签，例如：merchant_labels_2021-11.csv）
    output_dir = '../data/processed/labels_per_snapshot'
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    df_fraud = load_fraud_labels(fraud_labels_path)
    merchant_id_map = load_merchant_id_map(merchant_id_map_path)

    # 定义需要处理的时间截面（可根据图文件名自动提取，或者手工指定）
    snapshot_list = ['2021-11', '2021-12', '2022-01']  # 根据实际情况调整

    for snapshot in snapshot_list:
        df_labels = construct_labels_for_snapshot(df_fraud, merchant_id_map, snapshot)
        output_path = os.path.join(output_dir, f'merchant_labels_{snapshot}.csv')
        df_labels.to_csv(output_path, index=False)
        print(f"已保存时间截面 {snapshot} 的商户标签到 {output_path}")


if __name__ == "__main__":
    main()