# src/generate_id_maps.py

import os
import pandas as pd
import pickle
import logging
from utils import setup_logging

def generate_id_maps(processed_dir='../data/processed/'):
    """
    从所有时间截面的CSV文件中收集唯一的商户ID和商品名称，然后生成并保存ID映射。
    """
    snapshot_dir = os.path.join(processed_dir, 'time_snapshots')
    merchants = set()
    goods_set = set()

    # 遍历所有时间截面的CSV文件，收集商户ID和商品名称
    for snapshot_file in os.listdir(snapshot_dir):
        if snapshot_file.endswith('.csv'):
            snapshot_path = os.path.join(snapshot_dir, snapshot_file)
            try:
                df = pd.read_csv(snapshot_path, usecols=['LICEN_NO', 'GOODSNAME'], dtype={'LICEN_NO': str, 'GOODSNAME': str})
                # 清洗数据，去除空值和前后空白字符
                df['LICEN_NO'] = df['LICEN_NO'].str.strip().dropna()
                df['GOODSNAME'] = df['GOODSNAME'].str.strip().dropna()
                merchants.update(df['LICEN_NO'].unique())
                goods_set.update(df['GOODSNAME'].unique())
            except KeyError as e:
                logging.error(f"文件 {snapshot_path} 缺少必要的列: {e}")
                continue
            except Exception as e:
                logging.error(f"读取文件 {snapshot_path} 时出错：{e}")
                continue

    merchants = sorted(merchants)
    goods = sorted(goods_set)

    # 创建ID映射，确保键为字符串，值为整数
    merchant_id_map = {str(comid): int(idx) for idx, comid in enumerate(merchants)}
    goods_id_map = {str(goods_name): int(idx) for idx, goods_name in enumerate(goods)}

    # 添加断言确保映射类型正确
    assert all(isinstance(k, str) for k in merchant_id_map.keys()), "商户ID映射的键必须为字符串！"
    assert all(isinstance(v, int) for v in merchant_id_map.values()), "商户ID映射的值必须为整数！"
    assert all(isinstance(k, str) for k in goods_id_map.keys()), "商品名称映射的键必须为字符串！"
    assert all(isinstance(v, int) for v in goods_id_map.values()), "商品名称映射的值必须为整数！"

    # 保存ID映射为pickle文件
    with open(os.path.join(processed_dir, 'merchant_id_map.pkl'), 'wb') as f:
        pickle.dump(merchant_id_map, f)
    with open(os.path.join(processed_dir, 'goods_id_map.pkl'), 'wb') as f:
        pickle.dump(goods_id_map, f)

    logging.info(
        f"生成并保存merchant_id_map.pkl和goods_id_map.pkl，共{len(merchant_id_map)}个商户，{len(goods_id_map)}个商品。")

def main():
    # 设置日志
    setup_logging(log_file='logs/generate_id_maps.log')
    logger = logging.getLogger()
    logger.info("开始生成ID映射。")

    processed_dir = '../data/processed/'
    generate_id_maps(processed_dir)
    logger.info("ID映射生成完成。")

if __name__ == "__main__":
    main()