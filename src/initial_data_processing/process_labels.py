import pandas as pd
import torch

# -----------------------------
# 2. 处理违规整体标签，不考虑时间区间
# -----------------------------

# 违规记录的 Excel 文件路径
excel_path = "/data/raw/illicitInfo.xlsx"

# 读取 Excel 数据
df = pd.read_excel(excel_path)

def clean_merchant_id(x):
    """
    清洗商户编号：若是数字或以 '.0' 结尾的字符串，则转换为整数后再转为字符串。
    否则去除两端空白。
    """
    if pd.isna(x):
        return None
    # 如果 x 是数字，则转换为 int 后再转成字符串
    if isinstance(x, float):
        return str(int(x))
    elif isinstance(x, str):
        s = x.strip()
        if s.endswith('.0'):
            try:
                return str(int(float(s)))
            except Exception:
                return s
        return s
    else:
        return str(x)

# 清洗 'LICEN_NO' 列，确保每个商户编号不会带有 .0 后缀
df['LICEN_NO'] = df['LICEN_NO'].apply(clean_merchant_id)
# 注意：如果还有其他需要清洗的情况，可在 clean_merchant_id 中增加逻辑

# 构造整体违规标签：只要该文件中出现的商户，标签设为 1
merchant_overall_labels = {}
for merchant_id in df['LICEN_NO'].dropna().unique():
    merchant_overall_labels[merchant_id] = 1

# 示例打印部分商户的整体违规标签
print("部分商户的整体违规标签：")
for merchant_id, label in list(merchant_overall_labels.items())[:5]:
    print(f"商户 {merchant_id} : label = {label}")

# 保存整体标签到文件
output_label_path = "/data/processed/merchant_overall_labels.pt"
torch.save(merchant_overall_labels, output_label_path)
print(f"商户整体违规标签已保存到 {output_label_path}")