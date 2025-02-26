#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
该脚本用于从月度订单数据中提取详细描述的文本 prompt。
订单 CSV 文件中包含结构化的订购信息，例如 COMID, LICEN_NO, QTY, PRICE, AMT,
BORN_DATE, CONFORM_DATE, GOODSNAME, PACKBAK。

流程说明：
1. 对于每个订单文件，假定其中所有订单属于同一个月（也可通过文件名判断）。
2. 根据商户编号 LICEN_NO 对订单做分组：
   - 统计总下单次数（订单数）、总订购量、总金额以及平均单价
   - 对同一商户内，根据不同的商品 GOODSNAME 再做一个细粒度的统计，计算：
     • 每种商品被订购多少次
     • 每种商品的总订购量
     • 每种商品的平均单价（总金额/总数量）
3. 根据这些统计信息生成一段描述性文本，文本中详细描述了该商户本月的整体订单状况以及各个商品的详细情况。
4. 最后附加任务说明（例如要求下游模型生成时序特征，这部分说明可根据实际需要调整）。
5. 将提取结果保存为 CSV 文件，文件名格式如：detailed_prompts_YYYY-MM.csv

示例输出：
"In 2022-10, Merchant 410101335907 had 10 orders with a total quantity of 28.00 units and total amount $2698.80. The average unit price was $96.39.
Detailed breakdown by product: For 黄金叶(爱尚): ordered 2 times, total quantity 2.00 units, average price $139.92; For 黄金叶(小目标): ordered 2 times, total quantity 2.00 units, average price $121.90; For 黄金叶(商鼎): ordered 2 times, total quantity 2.00 units, average price $175.00; ...
Task: Please analyze the above data and generate enhanced temporal features for prediction.
Output your answer in JSON format as: { 'trend_score': <number>, 'volatility': <number>, 'growth_probability': <number> }."
"""

import os
import pandas as pd
import numpy as np
from dateutil.parser import parse  # 用于解析日期字符串（如果需要）

def extract_detailed_prompts_for_file(file_path):
    """
    对单个订单 CSV 文件进行处理，按商户（LICEN_NO）分组生成详细的描述性文本 prompt。

    参数：
        file_path (str): CSV 文件路径，例如 "orders_2021-11.csv"

    返回：
        List[dict]: 每个字典包含字段：
            - month: 订单所在月份（例如 "2022-10"），可以由文件名或者 CONFORM_DATE 得到
            - merchant_id: 商户编号
            - prompt: 根据模板生成的详细文本描述
    """
    try:
        # 读取 CSV 文件，并强制将 LICEN_NO、GOODSNAME 解析为字符串
        df = pd.read_csv(file_path, dtype={'LICEN_NO': str, 'GOODSNAME': str}, low_memory=False)
    except Exception as e:
        print(f"加载文件 {file_path} 失败: {e}")
        return []

    if df.empty:
        return []

    # 尝试从 CONFORM_DATE 字段中解析月份，如果解析失败，可按文件命名确定
    try:
        first_date = str(df['CONFORM_DATE'].iloc[0])
        parsed_date = parse(first_date)
        month = parsed_date.strftime("%Y-%m")
    except Exception as e:
        print(f"解析 CONFORM_DATE 失败: {e}")
        month = "Unknown"

    # 将数值字段转换为 float（防止字符串问题）
    for col in ['QTY', 'PRICE', 'AMT']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    prompts = []
    merchant_groups = df.groupby('LICEN_NO')

    for merchant, group in merchant_groups:
        total_orders = len(group)
        total_qty = group['QTY'].sum()
        total_amt = group['AMT'].sum()
        # 计算平均价格：使用总金额/总数量，如果数量为0则用字段均值
        if total_qty > 0:
            avg_price = total_amt / total_qty
        else:
            avg_price = group['PRICE'].mean()

        # 生成整体描述
        overview = (f"In {month}, Merchant {merchant} had {total_orders} orders with a total quantity of "
                    f"{total_qty:.2f} units and total amount of ${total_amt:.2f}. The average unit price was "
                    f"${avg_price:.2f}.")

        # 对每种商品进行详细统计
        breakdown_list = []
        goods_groups = group.groupby('GOODSNAME')
        for goods, sub_df in goods_groups:
            goods_orders = len(sub_df)
            goods_qty = sub_df['QTY'].sum()
            goods_amt = sub_df['AMT'].sum()
            if goods_qty > 0:
                goods_avg_price = goods_amt / goods_qty
            else:
                goods_avg_price = sub_df['PRICE'].mean()
            breakdown_list.append(
                f"For {goods}: ordered {goods_orders} times, total quantity {goods_qty:.2f} units, "
                f"average price ${goods_avg_price:.2f}"
            )
        breakdown = "; ".join(breakdown_list)

        # 拼接最终的 prompt 文本，并附加任务说明
        prompt_text = (f"{overview} Detailed breakdown by product: {breakdown}. "
                       "Task: Please analyze the above data and generate enhanced temporal features for prediction. "
                       "Output your answer in JSON format as: { 'trend_score': <number>, 'volatility': <number>, "
                       "'growth_probability': <number> }.")

        prompts.append({
            'month': month,
            'merchant_id': merchant,
            'prompt': prompt_text
        })

    return prompts


def extract_all_detailed_prompts(orders_dir, prompt_output_dir):
    """
    遍历订单 CSV 文件目录，并为每个月生成一个详细描述的 prompt CSV 文件，文件名沿用原始 orders 文件的时间戳。

    参数：
        orders_dir (str): 存放订单 CSV 文件的目录，例如 "data/processed/time_snapshots"
        prompt_output_dir (str): 存放生成的 prompt CSV 文件的目录，例如 "data/processed/prompts"
    """
    if not os.path.exists(prompt_output_dir):
        os.makedirs(prompt_output_dir)

    for file in sorted(os.listdir(orders_dir)):
        if file.startswith("orders_") and file.endswith(".csv"):
            file_path = os.path.join(orders_dir, file)
            prompts = extract_detailed_prompts_for_file(file_path)
            if not prompts:
                continue
            df_prompts = pd.DataFrame(prompts)
            # 直接从文件名提取时间戳部分，假设文件命名格式为 "orders_YYYY-MM.csv"
            try:
                month_from_filename = file.split('_')[1].replace(".csv", "")
            except Exception as e:
                print(f"解析文件 {file} 的月份失败: {e}")
                month_from_filename = "unknown"
            output_file = os.path.join(prompt_output_dir, f"detailed_prompts_{month_from_filename}.csv")
            df_prompts.to_csv(output_file, index=False)
            print(f"已生成 {output_file}，商户数量: {len(df_prompts)}")


def main():
    # 订单数据存放目录，请根据实际情况调整路径
    orders_dir = "/mnt/hdd/user4data/time_snapshots"
    prompt_output_dir = "/mnt/hdd/user4data/prompts"
    extract_all_detailed_prompts(orders_dir, prompt_output_dir)


if __name__ == "__main__":
    main()