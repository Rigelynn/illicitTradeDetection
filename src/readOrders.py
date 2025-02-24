import pandas as pd

# 定义文件路径
file_path = '../data/processed/time_snapshots/orders_2021-11.csv'  # 请根据实际路径修改

# 读取 CSV 文件的前20行
try:
    df = pd.read_csv(file_path, nrows=20)
    print("DataFrame 的列名：")
    print(df.columns.tolist())
    print("\nDataFrame 的数据类型：")
    print(df.dtypes)
    print("\n前20行数据预览：")
    print(df.to_string())  # 使用 to_string() 显示所有列
except FileNotFoundError:
    print(f"文件未找到：{file_path}")
except pd.errors.EmptyDataError:
    print("文件是空的。")
except pd.errors.ParserError:
    print("文件解析时出错。")
except Exception as e:
    print(f"读取文件时发生错误：{e}")