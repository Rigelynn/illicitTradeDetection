# 文件：build_global_mappings.py
import os
import pandas as pd
import pickle


def collect_all_regions(snapshot_dir):
    """收集所有时间截面的REGION值"""
    all_regions = set()
    for filename in os.listdir(snapshot_dir):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(snapshot_dir, filename), usecols=['LICEN_NO'])
            df['REGION'] = df['LICEN_NO'].astype(str).str[:6]
            all_regions.update(df['REGION'].unique())
    return sorted(all_regions)


def main():
    snapshot_dir = '../data/processed/time_snapshots'
    all_regions = collect_all_regions(snapshot_dir)

    # 生成region_id_map: {'REGION_VALUE': index}
    region_id_map = {region: idx for idx, region in enumerate(all_regions)}

    with open('../data/processed/region_id_map.pkl', 'wb') as f:
        pickle.dump(region_id_map, f)


if __name__ == "__main__":
    main()