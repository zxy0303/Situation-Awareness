import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

def step1_load_and_feature_engineer(filepath, trip_split_minutes=10, window_seconds=60, step_seconds=10):
    """
    1. 加载数据并按 'collectTime' 排序。
    2. 从第14列开始，对所有数值传感器进行滑动窗口统计。
    """
    print("--- 步骤 1: 开始加载数据... ---")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        print(f"加载 '{filepath}' 失败: {e}")
        return None

    print("正在转换时间戳并按 'collectTime' 排序...")
    df['collectTime'] = pd.to_datetime(df['collectTime'], unit='ms')
    df = df.sort_values('collectTime').reset_index(drop=True)
    sensor_cols = df.iloc[:, 13:].select_dtypes(include='number').columns
    print(f"已识别出 {len(sensor_cols)} 个数值型传感器特征（从第14列开始）。")

    print("正在按时间差（>{trip_split_minutes}分钟）拆分行程...")
    time_diff = df['collectTime'].diff().dt.total_seconds() / 60
    df['trip_id'] = (time_diff > trip_split_minutes).cumsum()

    window_features_list = []

    for trip_id, trip_data in df.groupby('trip_id'):
        if 'vehicleSpeed' in trip_data.columns and trip_data['vehicleSpeed'].max() == 0:
            continue

        trip_data = trip_data.reset_index(drop=True)
        start_time = trip_data['collectTime'].min()
        end_time = trip_data['collectTime'].max()

        print(f"\n--- 步骤 2: 正在处理行程 {trip_id} (时长: {end_time - start_time}) ---")

        current_time = start_time
        pbar_total = (end_time - start_time).total_seconds()
        if pbar_total <= 0:
            continue
        pbar = tqdm(total=pbar_total, desc=f"行程 {trip_id}")

        while current_time + pd.Timedelta(seconds=window_seconds) <= end_time:
            window_start = current_time
            window_end = current_time + pd.Timedelta(seconds=window_seconds)
            window_df = trip_data[(trip_data['collectTime'] >= window_start) &
                                  (trip_data['collectTime'] < window_end)]

            if len(window_df) > 5:
                stats = window_df[sensor_cols].agg(['mean', 'std', 'max', 'min'])
                flat_stats = stats.unstack()
                flat_stats.index = [f"{col}_{stat}" for col, stat in flat_stats.index]
                features_row_dict = flat_stats.to_dict()
                features_row_dict['window_start_time'] = window_start
                window_features_list.append(features_row_dict)

            current_time += pd.Timedelta(seconds=step_seconds)
            pbar.update(step_seconds)
        pbar.close()

    if not window_features_list:
        print("特征工程完成，但未提取到任何有效的驾驶窗口。")
        return None

    print(f"\n特征工程完成，共从所有行程中提取出 {len(window_features_list)} 个有效的时间窗口。")
    features_df = pd.DataFrame(window_features_list)
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)
    print("已使用 fillna(0) 填充计算失败(NaN)的特征。")

    return features_df

if __name__ == '__main__':

    INPUT_FILEPATH = '2025-08-23_processed_Q10-Q90.csv'
    OUTPUT_FEATURES_FILE = 'driving_features.parquet'

    print(f"--- 开始特征工程 ---")
    print(f"原始数据: {INPUT_FILEPATH}")

    features_df = step1_load_and_feature_engineer(
        INPUT_FILEPATH,
        trip_split_minutes=10,
        window_seconds=60,
        step_seconds=10
    )

    if features_df is not None:
        try:
            print(f"\n正在保存 {len(features_df)} 个特征窗口到 {OUTPUT_FEATURES_FILE}...")
            features_df.to_parquet(OUTPUT_FEATURES_FILE, index=False, engine='pyarrow')
            print("--- 特征工程文件保存完毕 ---")
        except ImportError:
            print("\n错误: 'pyarrow' 库未找到。")
            print("请运行 'pip install pyarrow' 来支持 Parquet 文件格式，然后重试。")
            # 备选方案：保存为 CSV (速度慢，不推荐)
            # print("正在尝试保存为 CSV (速度较慢)...")
            # features_df.to_csv('driving_features.csv', index=False)
            # print("--- 特征工程文件已保存为 CSV ---")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    else:

        print("未能从数据中提取任何特征，程序终止。")
