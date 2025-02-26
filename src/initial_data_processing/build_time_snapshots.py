# src/build_time_snapshots.py

import sys
import os
import logging
import pandas as pd
import pickle
from utils import setup_logging  # 确保utils.py在同一目录下或在sys.path中

def create_time_snapshots(df_orders, time_granularity='M', snapshot_dir='data/time_snapshots'):
    """
    按时间截面划分订单数据，并保存每个截面的数据。

    参数：
    - df_orders: DataFrame，预处理后的订单数据
    - time_granularity: str，时间截面粒度，如 'D'、'M'、'Q'
    - snapshot_dir: str，保存时间截面数据的目录

    返回：
    - snapshots: list，时间截面标签
    """
    df_orders['time_snapshot'] = df_orders['CONFORM_DATE'].dt.to_period(time_granularity)
    unique_snapshots = sorted(df_orders['time_snapshot'].unique())
    logging.info(f"共找到 {len(unique_snapshots)} 个时间截面。")

    # 创建目录
    os.makedirs(snapshot_dir, exist_ok=True)

    # 保存每个时间截面的数据
    snapshots = []
    for snap in unique_snapshots:
        df_snap = df_orders[df_orders['time_snapshot'] == snap]
        snap_str = str(snap)
        snap_path = os.path.join(snapshot_dir, f'orders_{snap_str}.csv')
        df_snap.to_csv(snap_path, index=False)
        snapshots.append(snap_str)
        logging.info(f"已保存时间截面 {snap_str}，记录数：{df_snap.shape[0]}")

    # 保存时间截面标签
    with open(os.path.join(snapshot_dir, 'snapshots.pkl'), 'wb') as f:
        pickle.dump(snapshots, f)
    logging.info("时间截面划分完成并已保存。")

    return snapshots

def main():
    # 设置日志
    setup_logging(log_file='logs/build_time_snapshots.log')
    logger = logging.getLogger()
    logger.info("开始创建时间截面。")

    # 定义数据目录
    processed_dir = '../../data/processed/'

    # 加载预处理后的订单数据
    try:
        orders_path = os.path.join(processed_dir, 'ordering_processed.csv')
        df_orders = pd.read_csv(orders_path, parse_dates=['BORN_DATE', 'CONFORM_DATE'])
        logger.info(f"成功加载订单数据，记录数：{df_orders.shape[0]}")
    except Exception as e:
        logger.error(f"加载订单数据失败：{e}")
        sys.exit(1)

    # 定义时间截面粒度
    time_granularity = 'M'  # 按月，可以修改为 'D' 按日或 'Q' 按季度

    # 创建时间截面
    snapshots = create_time_snapshots(df_orders, time_granularity=time_granularity, snapshot_dir=os.path.join(processed_dir, 'time_snapshots'))

    logger.info("时间截面创建完成。")

if __name__ == "__main__":
    main()