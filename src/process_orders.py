import pandas as pd
from sqlalchemy import create_engine
import os
import logging
import sys
from utils import setup_logging

def load_ordering_data(db_params, tables=['zz_linshi1', 'zz_linshi2']):
    """
    从多个MySQL表中加载订单数据并合并为一个DataFrame。
    """
    try:
        # 创建连接字符串
        connection_string = f"mysql+pymysql://{db_params['user']}:{db_params['password']}@" \
                            f"{db_params['host']}/{db_params['database']}?charset={db_params['charset']}"
        # 创建SQLAlchemy引擎
        engine = create_engine(connection_string, pool_recycle=3600)
        logging.info("成功创建SQLAlchemy引擎。")
    except Exception as e:
        logging.error(f"创建SQLAlchemy引擎失败：{e}")
        sys.exit(1)

    df_list = []
    for table in tables:
        logging.info(f"加载表 {table}...")
        try:
            df = pd.read_sql_query(f"SELECT * FROM `{table}`", engine)
            df_list.append(df)
            logging.info(f"已加载表 {table}，记录数：{df.shape[0]}")
        except Exception as e:
            logging.error(f"加载表 {table} 时发生错误：{e}")
            continue  # 可以选择是否在此中断脚本

    if not df_list:
        logging.error("未能加载任何表的数据。脚本将终止。")
        sys.exit(1)

    combined_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"成功合并了 {len(tables)} 个表的数据，共计 {combined_df.shape[0]} 条订单记录。")
    return combined_df

def preprocess_ordering_data(df_ordering):
    """
    预处理订单数据，包括处理缺失值、转换日期格式和提取省市区信息及时间特征。
    """
    initial_count = df_ordering.shape[0]
    # 处理缺失值
    df_ordering = df_ordering.dropna()
    after_dropna_count = df_ordering.shape[0]
    logging.info(f"删除缺失值后的订单数据量：从 {initial_count} 降至 {after_dropna_count} 条。")

    # 数据类型转换
    df_ordering['BORN_DATE'] = pd.to_datetime(df_ordering['BORN_DATE'], format='%Y%m%d', errors='coerce')
    df_ordering['CONFORM_DATE'] = pd.to_datetime(df_ordering['CONFORM_DATE'], format='%Y%m%d', errors='coerce')

    # 检查日期转换后的缺失值
    missing_born_date = df_ordering['BORN_DATE'].isna().sum()
    missing_conform_date = df_ordering['CONFORM_DATE'].isna().sum()
    if missing_born_date > 0 or missing_conform_date > 0:
        logging.warning(
            f"BORN_DATE 有 {missing_born_date} 个无效日期，CONFORM_DATE 有 {missing_conform_date} 个无效日期。")
        # 根据需要决定是否删除这些行
        df_ordering = df_ordering.dropna(subset=['BORN_DATE', 'CONFORM_DATE'])
        logging.info(f"删除无效日期后的订单数据量：{df_ordering.shape[0]} 条。")

    # 提取省市区信息
    df_ordering['PROVINCE'] = df_ordering['LICEN_NO'].astype(str).str[:2]
    df_ordering['CITY'] = df_ordering['LICEN_NO'].astype(str).str[2:4]
    df_ordering['DISTRICT'] = df_ordering['LICEN_NO'].astype(str).str[4:6]

    # 提取更多时间特征
    df_ordering['BORN_YEAR'] = df_ordering['BORN_DATE'].dt.year
    df_ordering['BORN_MONTH'] = df_ordering['BORN_DATE'].dt.month
    df_ordering['BORN_DAY'] = df_ordering['BORN_DATE'].dt.day
    df_ordering['BORN_WEEKDAY'] = df_ordering['BORN_DATE'].dt.weekday  # 0=Monday, 6=Sunday

    df_ordering['CONFORM_YEAR'] = df_ordering['CONFORM_DATE'].dt.year
    df_ordering['CONFORM_MONTH'] = df_ordering['CONFORM_DATE'].dt.month
    df_ordering['CONFORM_DAY'] = df_ordering['CONFORM_DATE'].dt.day
    df_ordering['CONFORM_WEEKDAY'] = df_ordering['CONFORM_DATE'].dt.weekday

    # 打印预处理后的数据摘要
    logging.info("预处理后的订单数据摘要：")
    logging.info(df_ordering.describe(include='all'))
    return df_ordering

def save_preprocessed_orders(df_orders, processed_dir):
    """
    保存预处理后的订单数据为CSV文件。
    """
    try:
        os.makedirs(processed_dir, exist_ok=True)
        orders_path = os.path.join(processed_dir, 'ordering_processed.csv')
        df_orders.to_csv(orders_path, index=False)
        logging.info(f"预处理后的订单数据已保存到 {orders_path}")
    except Exception as e:
        logging.error(f"保存预处理数据时发生错误：{e}")
        sys.exit(1)

def main():
    setup_logging()
    logging.info("订单数据处理脚本启动。")

    # 定义数据目录
    processed_dir = '../data/processed/'

    # 数据库连接参数（请根据实际情况填写）
    db_params = {
        'host': 'localhost',
        'user': 'zhenglin',
        'password': 'password',  # 替换为实际密码
        'database': 'HNYCZMC',
        'charset': 'utf8mb4'
    }

    # 加载原始订单数据
    logging.info("加载订单数据...")
    df_ordering = load_ordering_data(db_params, tables=['zz_linshi1', 'zz_linshi2'])

    # 预处理订单数据
    logging.info("预处理订单数据...")
    df_ordering = preprocess_ordering_data(df_ordering)

    # 保存预处理后的订单数据
    logging.info("保存预处理后的订单数据...")
    save_preprocessed_orders(df_ordering, processed_dir)

    logging.info("订单数据处理完成。")

if __name__ == "__main__":
    main()