# build_graphs_from_snapshots.py

import os
import sys
import logging
import pandas as pd
import dgl
import pickle
from build_hetero_graph_with_time import build_heterogeneous_graph_with_time

def load_id_maps(processed_dir):
    try:
        with open(os.path.join(processed_dir, 'merchant_id_map.pkl'), 'rb') as f:
            merchant_id_map = pickle.load(f)
        with open(os.path.join(processed_dir, 'goods_id_map.pkl'), 'rb') as f:
            goods_id_map = pickle.load(f)
        logging.info("成功加载商户和商品ID映射。")
        return merchant_id_map, goods_id_map
    except Exception as e:
        logging.error(f"加载ID映射失败：{e}")
        raise

def build_graphs_from_snapshots(snapshot_dir, merchant_id_map, goods_id_map, output_dir='data/graphs', include_timestamp=False):
    os.makedirs(output_dir, exist_ok=True)
    for snapshot_file in os.listdir(snapshot_dir):
        if snapshot_file.endswith('.csv'):
            snapshot_str = snapshot_file.replace('orders_', '').replace('.csv', '')
            snapshot_path = os.path.join(snapshot_dir, snapshot_file)
            try:
                df_snapshot = pd.read_csv(
                    snapshot_path,
                    parse_dates=['CONFORM_DATE', 'BORN_DATE'],
                    dtype={
                        'LICEN_NO': str,
                        'GOODSNAME': str,
                        'AMT': float,
                        'PROVINCE': str,
                        'CITY': str,
                        'DISTRICT': str,
                        'BORN_YEAR': 'Int64',
                        'BORN_MONTH': 'Int64',
                        'BORN_DAY': 'Int64',
                        'BORN_WEEKDAY': 'Int64',
                        'CONFORM_YEAR': 'Int64',
                        'CONFORM_MONTH': 'Int64',
                        'CONFORM_DAY': 'Int64',
                        'CONFORM_WEEKDAY': 'Int64'
                    },
                    low_memory=False
                )
                # 清洗关键字段
                df_snapshot['LICEN_NO'] = df_snapshot['LICEN_NO'].str.strip()
                df_snapshot['GOODSNAME'] = df_snapshot['GOODSNAME'].str.strip()
                df_snapshot.fillna({
                    'LICEN_NO': 'unknown',
                    'GOODSNAME': 'unknown',
                    'AMT': 0.0,
                    'PROVINCE': 'unknown',
                    'CITY': 'unknown',
                    'DISTRICT': 'unknown',
                    'BORN_YEAR': 0,
                    'BORN_MONTH': 0,
                    'BORN_DAY': 0,
                    'BORN_WEEKDAY': 0,
                    'CONFORM_YEAR': 0,
                    'CONFORM_MONTH': 0,
                    'CONFORM_DAY': 0,
                    'CONFORM_WEEKDAY': 0
                }, inplace=True)

                # 断言：确保关键字段没有缺失值
                assert not df_snapshot[['LICEN_NO', 'GOODSNAME']].isnull().any().any(), "LICEN_NO 和 GOODSNAME 不能有缺失值！"
            except Exception as e:
                logging.error(f"读取文件 {snapshot_path} 时出错：{e}")
                continue

            try:
                hetero_graph = build_heterogeneous_graph_with_time(
                    df_snapshot,
                    merchant_id_map,
                    goods_id_map,
                    include_timestamp=include_timestamp
                )
            except Exception as e:
                logging.error(f"构建异构图时出错：{e}")
                continue

            if hetero_graph:
                graph_path = os.path.join(output_dir, f'hetero_graph_{snapshot_str}.dgl')
                try:
                    dgl.save_graphs(graph_path, [hetero_graph])
                    logging.info(f"保存时间截面 {snapshot_str} 的异构图到 {graph_path}")
                except Exception as e:
                    logging.error(f"保存异构图失败：{e}")
            else:
                logging.warning(f"时间截面 {snapshot_str} 的异构图为空，未保存。")

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    processed_dir = '../data/processed/'
    snapshot_dir = os.path.join(processed_dir, 'time_snapshots')
    output_dir = os.path.join(processed_dir, 'graphs')

    if not os.path.exists(snapshot_dir):
        logging.error(f"时间截面目录未找到: {snapshot_dir}")
        sys.exit(1)

    try:
        merchant_id_map, goods_id_map = load_id_maps(processed_dir)
    except Exception as e:
        logging.error(f"加载ID映射失败: {e}")
        sys.exit(1)

    # 这里设定为包含节点特征（如地理信息）
    build_graphs_from_snapshots(snapshot_dir, merchant_id_map, goods_id_map, output_dir=output_dir, include_timestamp=False)
    logging.info("所有时间截面的异构图已构建完成。")

if __name__ == "__main__":
    main()