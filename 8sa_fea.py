import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np


def run_structured_correlation():
    print("--- 开始运行：针对性相关性分析 ---")

    # 1. 加载数据和配置文件
    csv_path = 'labeled_source_data_with_situations.csv'
    json_path = 'selected_features_structured.json'

    try:
        df = pd.read_csv(csv_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        return

    # 2. 建立映射关系
    situation_feature_map = {}
    for category in structured_data:
        for sub_cat in category['sub_categories']:
            sub_name = sub_cat['sub_category_name']
            features = sub_cat['features']
            situation_feature_map[sub_name] = features

    # 3. 识别 CSV 中的态势列
    situation_cols_csv = df.columns[-8:].tolist()
    col_to_json_key = {}
    for col in situation_cols_csv:
        for key in situation_feature_map.keys():
            if col.endswith(key):
                col_to_json_key[col] = key
                break

    # 4. 计算相关性
    all_relevant_features = set()
    for feats in situation_feature_map.values():
        all_relevant_features.update(feats)

    results_df = pd.DataFrame(index=list(all_relevant_features), columns=situation_cols_csv)

    for sit_col in situation_cols_csv:
        json_key = col_to_json_key.get(sit_col)
        if not json_key: continue

        target_features = situation_feature_map[json_key]
        valid_features = [f for f in target_features if f in df.columns]
        if not valid_features: continue

        temp_df = df[valid_features + [sit_col]].copy()

        # 态势标签数值化
        if temp_df[sit_col].dtype == 'object':
            temp_df[sit_col] = temp_df[sit_col].fillna('Unknown').astype(str)
            le = LabelEncoder()
            temp_df[sit_col] = le.fit_transform(temp_df[sit_col])
        else:
            temp_df[sit_col] = temp_df[sit_col].fillna(0)

        # 特征数值化
        for f in valid_features:
            temp_df[f] = pd.to_numeric(temp_df[f], errors='coerce')

        if temp_df[sit_col].nunique() > 1:
            corr_matrix = temp_df.corr()
            results_df.loc[valid_features, sit_col] = corr_matrix[sit_col].drop(sit_col)

    # 5. 绘制并保存热图
    results_df_clean = results_df.dropna(how='all').astype(float)

    plt.figure(figsize=(12, 10))
    sns.heatmap(results_df_clean, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, center=0)
    plt.title('Correlation Heatmap (Specified Features)')
    plt.xlabel('Situations')
    plt.ylabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('structured_correlation_heatmap.png')
    print("结果已保存: structured_correlation_heatmap.png")


if __name__ == "__main__":
    # 设置绘图风格和字体
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    run_structured_correlation()