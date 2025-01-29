# src/data_processing.py

import pandas as pd
import psycopg2
from psycopg2 import sql
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import dgl
import torch


def load_ordering_data(sql_file_path, db_params):
    """加载订单数据"""
    conn = psycopg2.connect(**db_params)
    df = pd.read_sql_query("SELECT * FROM zz_linshi1", conn)
    conn.close()
    return df


def load_fraud_labels(excel_file_path):
    """加载违规标签"""
    df = pd.read_excel(excel_file_path)
    return df


def preprocess_ordering_data(df_ordering):
    """订单数据预处理"""
    # 处理缺失值
    df_ordering.dropna(inplace=True)

    # 数据类型转换
    df_ordering['BORN_DATE'] = pd.to_datetime(df_ordering['BORN_DATE'], format='%Y%m%d')
    df_ordering['CONFORM_DATE'] = pd.to_datetime(df_ordering['CONFORM_DATE'], format='%Y%m%d')

    # 提取省市区信息
    df_ordering['PROVINCE'] = df_ordering['LICEN_NO'].astype(str).str[:2]
    df_ordering['CITY'] = df_ordering['LICEN_NO'].astype(str).str[2:4]
    df_ordering['DISTRICT'] = df_ordering['LICEN_NO'].astype(str).str[4:6]

    return df_ordering


def save_preprocessed_data(df, path):
    """保存预处理后的数据"""
    df.to_csv(path, index=False)


def main():
    data_dir = '../data/raw/'
    processed_dir = '../data/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    # 数据库连接参数（请根据实际情况填写）
    db_params = {
        'dbname': 'your_db',
        'user': 'your_user',
        'password': 'your_password',
        'host': 'your_host',
        'port': 'your_port'
    }

    # 加载原始订单数据
    ordering_sql_file = os.path.join(data_dir, 'file1.sql')  # 实际文件名
    df_ordering = load_ordering_data(ordering_sql_file, db_params)

    # 加载违规标签
    fraud_excel_file = os.path.join(data_dir, 'fraud_labels.xlsx')  # 实际文件名
    df_fraud = load_fraud_labels(fraud_excel_file)

    # 预处理订单数据
    df_ordering = preprocess_ordering_data(df_ordering)

    # 保存预处理后的数据
    save_preprocessed_data(df_ordering, os.path.join(processed_dir, 'ordering_processed.csv'))
    save_preprocessed_data(df_fraud, os.path.join(processed_dir, 'fraud_labels_processed.csv'))

    print("数据预处理完成。")


if __name__ == "__main__":
    main()