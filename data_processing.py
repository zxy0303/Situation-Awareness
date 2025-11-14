import pandas as pd
import numpy as np

# --- 第 1 步：加载和排序 ---
try:
    df = pd.read_csv('data/2025-08-23.csv', low_memory=False)
    print(f"原始数据已加载: {df.shape[0]} 行, {df.shape[1]} 列")

    # 转换 'collectTime' 为 datetime 对象并排序
    df['collectTime_dt'] = pd.to_datetime(df['collectTime'], unit='ms')
    df = df.set_index('collectTime_dt')
    df = df.sort_index()
    print("数据已按 'collectTime' 排序并设置索引。")
    print("---------------------------------")

except Exception as e:
    print(f"在加载或排序时发生错误: {e}")
    raise e
df = df[df['cycleType'] != 10000]
print("开始检查空缺值 > 50% 的列...")
nan_threshold = 0.5
nan_ratio = df.isnull().mean()
cols_to_drop = nan_ratio[nan_ratio > nan_threshold].index

if len(cols_to_drop) > 0:
    print(f"检测到 {len(cols_to_drop)} 列的空缺值 > {nan_threshold * 100}%，将予以忽略：")
    # 打印所有被删除的列
    for col in cols_to_drop:
        print(f"  - {col} (缺失比例: {nan_ratio[col]:.2%})")

    # 执行删除
    df = df.drop(columns=cols_to_drop)
    print(f"清理后数据形状: {df.shape}")
else:
    print(f"所有列的空缺值均未超过 {nan_threshold * 100}% 阈值。")
print("---------------------------------")
# --- 第 2 步：按 N-Unique=10 阈值分类列 ---
all_float_cols = df.select_dtypes(include=[np.number]).columns
nunique_threshold = 10

status_cols = []
numerical_cols_for_detection = []

for col in all_float_cols:
    num_unique = df[col].nunique()
    if num_unique > nunique_threshold:
        numerical_cols_for_detection.append(col)
    else:
        status_cols.append(col)

print(f"分类完成：")
print(f"  {len(status_cols)} 个状态列 (唯一值 <= {nunique_threshold})，将跳过异常检测。")
print(f"  {len(numerical_cols_for_detection)} 个数值列 (唯一值 > {nunique_threshold})，将进行异常检测。")
print("---------------------------------")

# --- 第 3 步：使用 Q10/Q90 和 3.0 乘数进行异常检测 ---

if not numerical_cols_for_detection:
    print("没有找到需要进行异常检测的数值列。跳过检测步骤。")
    df_cleaned = df

else:
    print(f"开始在 {len(numerical_cols_for_detection)} 个数值列上使用 Q10/Q90 (3.0 乘数) 规则检测异常值...")


    Q10 = df[numerical_cols_for_detection].quantile(0.10)
    Q90 = df[numerical_cols_for_detection].quantile(0.90)
    IDR = Q90 - Q10 

    multiplier = 3.0
    lower_bound = Q10 - multiplier * IDR
    upper_bound = Q90 + multiplier * IDR
    outlier_mask = (df[numerical_cols_for_detection] < lower_bound) | (df[numerical_cols_for_detection] > upper_bound)
    rows_with_outliers = outlier_mask.any(axis=1)
    outlier_count = rows_with_outliers.sum()
    print(f"检测完成：")
    print(f"  原始数据行数: {df.shape[0]}")
    df_cleaned = df[~rows_with_outliers]
    print(f"  已删除 {outlier_count} 行。")
    print(f"  剩余行数: {df_cleaned.shape[0]}")
    print("---------------------------------")


# --- 第 5 步：填充缺失值 (对所有 float 列) ---
print(f"开始对 {len(all_float_cols)} 个列（包括状态列和数值列）进行向前向后填充...")

float_cols_to_fill = df_cleaned.select_dtypes(include=[np.float64]).columns

df_filled = df_cleaned[float_cols_to_fill].ffill()
df_filled = df_filled.bfill()
df_cleaned[float_cols_to_fill] = df_filled

print("缺失值填充完成。")
print("---------------------------------")

# --- 第 6 步：保存文件 ---
output_filename = '2025-08-23_processed_Q10-Q90.csv'
try:
    df_cleaned.to_csv(output_filename, index=True, encoding='utf-8-sig')
    print(f"最终处理后的数据已成功保存到: {output_filename}")
    print(f"总共保存了 {df_cleaned.shape[0]} 行数据。")
except Exception as e:
    print(f"保存文件时发生错误: {e}")


