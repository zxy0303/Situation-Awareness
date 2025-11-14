import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# --- 1. 定义文件和输出目录 ---
JSON_FILE = 'selected_features_structured.json'
DATA_FILE = '2025-08-23_processed_Q10-Q90.csv'
OUTPUT_DIR = 'correlation_heatmaps'

# 确保 Matplotlib 可以显示中文
# (请确保您的环境中已安装支持中文的字体，例如 SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 创建输出目录 ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
print(f"将在此处保存热图: {OUTPUT_DIR}/")

# --- 3. 加载并解析 JSON 特征结构 ---
subdimension_features = {}
try:
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        structure = json.load(f)

    # 遍历类别 (例如 "Environment Dimension")
    for category in structure:
        # 遍历子类别
        for sub_cat in category.get('sub_categories', []):
            sub_name = sub_cat.get('sub_category_name')
            features = sub_cat.get('features', [])

            if sub_name and features:
                # 使用 set 来自动处理并合并重复的特征
                if sub_name not in subdimension_features:
                    subdimension_features[sub_name] = set()

                subdimension_features[sub_name].update(features)

    # 将 set 转换回 list
    for sub_name in subdimension_features:
        subdimension_features[sub_name] = list(subdimension_features[sub_name])

    print(f"\n成功从 {JSON_FILE} 加载了 {len(subdimension_features)} 个子维度。")
    print(f"子维度列表: {list(subdimension_features.keys())}")

except FileNotFoundError:
    print(f"错误: JSON 文件 {JSON_FILE} 未找到。")
    sys.exit(1)
except Exception as e:
    print(f"加载或解析 {JSON_FILE} 失败: {e}")
    sys.exit(1)

# --- 4. 加载主数据文件 ---
print(f"\n正在加载主数据文件: {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
    all_data_columns = set(df.columns)
    print(f"主数据加载成功，形状: {df.shape}")
except FileNotFoundError:
    print(f"错误: 数据文件 {DATA_FILE} 未找到。")
    sys.exit(1)
except Exception as e:
    print(f"加载 {DATA_FILE} 失败: {e}")
    sys.exit(1)

# --- 5. 遍历、分析和绘图 ---
print("\n--- 开始分析每个子维度的内部相关性 ---")

for sub_name, feature_list in subdimension_features.items():
    print(f"\n--- G正在处理子维度: {sub_name} ---")

    # 5.1 检查哪些特征在CSV中可用
    available_features = [f for f in feature_list if f in all_data_columns]
    missing_features = [f for f in feature_list if f not in all_data_columns]

    if missing_features:
        print(f"  注意: {len(missing_features)} 个特征在CSV中未找到: {missing_features}")

    # 5.2 检查是否有足够的特征进行分析
    if len(available_features) < 2:
        print(f"  跳过: 需要至少2个可用的特征来计算相关性，但只找到 {len(available_features)} 个。")
        continue

    print(f"  找到 {len(available_features)} 个可用特征进行分析。")

    # 5.3 创建子 DataFrame 并计算相关性
    sub_df = df[available_features]

    # (确保所有数据都是数值类型，以防万一)
    sub_df = sub_df.apply(pd.to_numeric, errors='coerce')

    # (删除全是NaN的列，以防 'coerce' 产生了问题)
    sub_df = sub_df.dropna(axis=1, how='all')

    if len(sub_df.columns) < 2:
        print(f"  跳过: 在转换为数值后，剩余的有效特征不足2个。")
        continue

    corr_matrix = sub_df.corr()

    # 5.4 打印相关性强度 (满足用户需求)
    print(f"  '{sub_name}' 的相关性矩阵:")
    # 使用 .to_string() 以便在日志中完整显示
    print(corr_matrix.to_string(float_format="%.2f"))

    # 5.5 绘制并保存热图
    try:
        # 动态调整画布大小
        plot_size = max(6, len(corr_matrix) * 0.7)
        plt.figure(figsize=(plot_size + 2, plot_size))

        # 创建一个遮罩，隐藏相关性矩阵的上三角（因为它是对称的）
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        heatmap = sns.heatmap(
            corr_matrix,
            mask=mask,  # 应用遮罩
            annot=True,  # 在单元格中显示数值
            fmt=".2f",  # 格式化数值为两位小数
            cmap='vlag',  # 使用蓝-白-红发散色图
            center=0,  # 色图以0为中心
            square=True,  # 强制单元格为方形
            linewidths=.5,
            cbar_kws={"shrink": .5}  # 缩小颜色条
        )

        plt.title(f"'{sub_name}' 子维度内部特征相关性", fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()  # 自动调整布局防止标签重叠

        # 5.6 清理文件名并保存
        safe_filename = "".join(c for c in sub_name if c.isalnum() or c in ('&', '_', '-')).rstrip()
        if not safe_filename:
            safe_filename = f"subdim_{np.random.randint(1000)}"  # 备用文件名

        output_path = os.path.join(OUTPUT_DIR, f"corr_heatmap_{safe_filename}.png")
        plt.savefig(output_path)
        plt.close()  # 关闭图形，释放内存

        print(f"  已保存热图到: {output_path}")

    except Exception as e:
        print(f"  !! 在为 '{sub_name}' 绘制热图时失败: {e}")

print("\n--- 所有子维度处理完毕 ---")