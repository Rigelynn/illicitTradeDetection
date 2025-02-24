# build_graphs_from_snapshots_pyg.py

import os
import sys
import logging
import pandas as pd
import pickle
import torch
from build_hetero_graph_with_time_pyg import build_pyg_hetero_graph

def setup_logging():
    """配置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

def load_id_maps(processed_dir):
    """加载商户、商品和区域的ID映射文件"""
    try:
        with open(os.path.join(processed_dir, 'merchant_id_map.pkl'), 'rb') as f:
            merchant_id_map = pickle.load(f)
        with open(os.path.join(processed_dir, 'goods_id_map.pkl'), 'rb') as f:
            goods_id_map = pickle.load(f)
        with open(os.path.join(processed_dir, 'region_id_map.pkl'), 'rb') as f:
            region_id_map = pickle.load(f)
        logging.info("成功加载商户、商品和区域ID映射。")
        return merchant_id_map, goods_id_map, region_id_map
    except Exception as e:
        logging.error(f"加载ID映射失败：{e}")
        raise

def process_snapshot_file(snapshot_file_path, merchant_id_map, goods_id_map, region_id_map, output_dir, include_timestamp=False):
    """
    处理单个时间截面的订单CSV文件，构造 PyG 异构图并保存

    参数：
        snapshot_file_path: str，订单数据 CSV 文件路径
        merchant_id_map    : dict，商户名称到ID的映射
        goods_id_map       : dict，商品名称到ID的映射
        region_id_map      : dict，区域代码到ID的映射
        output_dir         : str，保存构造后图的目录
        include_timestamp  : bool，是否添加时间戳特征
    """
    try:
        df_order = pd.read_csv(
            snapshot_file_path,
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
        logging.info(f"加载订单数据 {snapshot_file_path} 成功，包含 {len(df_order)} 条记录。")

        # 清洗关键字段
        df_order['LICEN_NO'] = df_order['LICEN_NO'].str.strip()
        df_order['GOODSNAME'] = df_order['GOODSNAME'].str.strip()
        df_order.fillna({
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
        assert not df_order[['LICEN_NO', 'GOODSNAME']].isnull().any().any(), "LICEN_NO 和 GOODSNAME 不能有缺失值！"
    except Exception as e:
        logging.error(f"读取文件 {snapshot_file_path} 时出错：{e}")
        return

    try:
        # 构建 PyG 异构图
        hetero_graph = build_pyg_hetero_graph(
            df_order,
            merchant_id_map,
            goods_id_map,
            region_id_map,
            include_timestamp=include_timestamp
        )
    except Exception as e:
        logging.error(f"构建异构图时出错：{e}")
        return

    # 提取 snapshot_str 以构造输出文件名
    snapshot_str = os.path.basename(snapshot_file_path).replace('orders_', '').replace('.csv', '')

    if hetero_graph:
        output_path = os.path.join(output_dir, f'hetero_graph_{snapshot_str}.pt')
        try:
            torch.save(hetero_graph, output_path)
            logging.info(f"保存时间截面 {snapshot_str} 的异构图到 {output_path}")
        except Exception as e:
            logging.error(f"保存异构图失败：{e}")
    else:
        logging.warning(f"时间截面 {snapshot_str} 的异构图为空，未保存。")

def build_graphs_from_snapshots(snapshot_dir, merchant_id_map, goods_id_map, region_id_map, output_dir='data/pyg_graphs', include_timestamp=False):
    """
    遍历 snapshot_dir 中的所有订单CSV文件，构建 PyG 异构图并保存。

    参数：
        snapshot_dir     : str，订单CSV文件所在目录
        merchant_id_map  : dict，商户名称到ID的映射
        goods_id_map     : dict，商品名称到ID的映射
        region_id_map    : dict，区域代码到ID的映射
        output_dir       : str，保存构造后图的目录
        include_timestamp: bool，是否添加时间戳特征
    """
    os.makedirs(output_dir, exist_ok=True)
    snapshot_files = [f for f in os.listdir(snapshot_dir) if f.endswith('.csv')]
    logging.info(f"找到 {len(snapshot_files)} 个订单数据文件需要处理。")

    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(snapshot_dir, snapshot_file)
        logging.info(f"开始处理图文件：{snapshot_path}")
        process_snapshot_file(snapshot_path, merchant_id_map, goods_id_map, region_id_map, output_dir, include_timestamp)

def main():
    setup_logging()

    # 定义路径
    processed_dir = '../data/processed/'
    snapshot_dir = os.path.join(processed_dir, 'time_snapshots')
    output_dir = os.path.join(processed_dir, 'pyg_graphs')  # 保存为 PyG 格式的图

    # 检查路径是否存在
    if not os.path.exists(snapshot_dir):
        logging.error(f"时间截面目录未找到: {snapshot_dir}")
        sys.exit(1)

    try:
        merchant_id_map, goods_id_map, region_id_map = load_id_maps(processed_dir)
    except Exception as e:
        logging.error(f"加载ID映射失败: {e}")
        sys.exit(1)

    # 构建 PyG 格式的图
    build_graphs_from_snapshots(snapshot_dir, merchant_id_map, goods_id_map, region_id_map, output_dir=output_dir, include_timestamp=False)
    logging.info("所有时间截面的 PyG 异构图已构建完成。")

if __name__ == "__main__":
    main()