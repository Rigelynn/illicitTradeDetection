import pandas as pd
import sqlite3
import os


def import_sql_to_csv(sql_file_path, table_name, output_csv_path):
    """
    将 SQL 文件中的表数据导出为 CSV。
    """
    conn = sqlite3.connect(sql_file_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    df.to_csv(output_csv_path, index=False)
    conn.close()


def clean_orders_data(orders_csv_path, output_csv_path):
    """
    清洗订单数据，处理缺失值和重复记录。
    """
    df = pd.read_csv(orders_csv_path)

    # 删除重复记录
    df.drop_duplicates(inplace=True)

    # 填充缺失值
    numerical_cols = ['QTY', 'PRICE', 'AMT']
    df[numerical_cols] = df[numerical_cols].fillna(0)

    categorical_cols = ['GOODSNAME', 'PACKBAK']
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # 标准化 LICEN_NO
    df['LICEN_NO'] = df['LICEN_NO'].str.upper()

    df.to_csv(output_csv_path, index=False)
    print(f"清洗后的数据已保存到 {output_csv_path}")


def main():
    raw_orders1_sql = "data/raw/orders1.sql"
    raw_orders2_sql = "data/raw/orders2.sql"
    fraud_labels_excel = "data/raw/fraud_labels.xlsx"
    processed_orders_csv = "data/processed/orders_cleaned.csv"

    # 转换 SQL 数据到 CSV
    print("导入订单1 SQL 数据到 CSV...")
    import_sql_to_csv(raw_orders1_sql, "orders", "data/raw/orders1.csv")

    print("导入订单2 SQL 数据到 CSV...")
    import_sql_to_csv(raw_orders2_sql, "orders", "data/raw/orders2.csv")

    # 合并订单数据
    print("合并订单数据...")
    orders1_df = pd.read_csv("data/raw/orders1.csv")
    orders2_df = pd.read_csv("data/raw/orders2.csv")
    combined_orders = pd.concat([orders1_df, orders2_df], ignore_index=True)

    # 清洗订单数据
    print("清洗订单数据...")
    clean_orders_data("data/raw/orders1.csv", processed_orders_csv)


if __name__ == "__main__":
    main()