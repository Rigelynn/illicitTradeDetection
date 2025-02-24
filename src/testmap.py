import os
import pickle


def check_goods_id_map_total(file_path='../data/processed/goods_id_map.pkl', onehot_threshold=1000):
    """
    加载 goods_id_map.pkl 文件并统计不同商品种类的数量，
    根据阈值 onehot_threshold 判断是否适合采用 one-hot 编码。

    参数:
        file_path: goods_id_map.pkl 文件路径
        onehot_threshold: 如果不同商品数小于此阈值，认为适合采用 one-hot 编码
    """
    # 加载 goods_id_map.pkl
    with open(file_path, 'rb') as f:
        goods_id_map = pickle.load(f)

    total_goods = len(goods_id_map)
    print(f"总共有 {total_goods} 个不同的商品。")

    # 判断是否适合做 one-hot 编码
    if total_goods <= onehot_threshold:
        print("商品种类数量较少，适合采用 one-hot 编码。")
    else:
        print("商品种类数量较多，建议使用其他编码方式（如嵌入向量）。")


if __name__ == '__main__':
    check_goods_id_map_total()