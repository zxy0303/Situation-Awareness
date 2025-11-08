import pandas as pd
import numpy as np
import json
import sys

# --- 1. 定义输入文件 ---
FEATURES_JSON_FILE = 'selected_features_structured.json'
CLUSTERED_DATA_FILE = 'clustered_driving_segments_MANUAL_CHOICE.csv'
OUTPUT_FILE = 'cluster_profiles_for_llm_FULL.csv'  # 新的输出文件名
PASS_THROUGH_FEATURES = [
    'SOC_drop_rate'
]


# --- 2. 从 JSON 加载并构建“关键特征列表” (包含所有4个统计) ---
def load_features_from_json(json_path):
    """
    读取 structured.json 文件，
    提取所有 feature_name，并为它们附加 '_mean', '_std', '_max', '_min' 后缀。
    """
    print(f"正在从 {json_path} 加载特征列表...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            structure = json.load(f)

        key_features_list = []
        base_feature_count = 0
        pass_through_count = 0

        # 遍历嵌套结构
        for big_cat in structure:
            for sub_cat in big_cat.get('sub_categories', []):
                for feature_name in sub_cat.get('features', []):
                    if feature_name in PASS_THROUGH_FEATURES:
                        key_features_list.append(feature_name)  # 1. 直接添加
                        pass_through_count += 1
                    else:
                        base_feature_count += 1
                        # 2. 附加所有4个统计后缀
                        key_features_list.append(f"{feature_name}_mean")
                        key_features_list.append(f"{feature_name}_std")
                        key_features_list.append(f"{feature_name}_max")
                        key_features_list.append(f"{feature_name}_min")

        print(f"从 JSON 中成功提取 {base_feature_count} 个基础特征。")
        print(f"已构建 {len(key_features_list)} 个统计特征（{base_feature_count} x 4）用于分析。")
        return key_features_list

    except FileNotFoundError:
        print(f"错误: 未找到特征定义文件 '{json_path}'。")
        return None
    except Exception as e:
        print(f"读取或解析 JSON 时出错: {e}")
        return None


# --- 3. 加载已聚类的数据 ---
def load_clustered_data(csv_path):
    try:
        print(f"正在加载已聚类的特征文件: {csv_path}...")
        df = pd.read_csv(csv_path, low_memory=False)

        if 'category_id' not in df.columns:
            print(f"错误: 文件 {csv_path} 中缺少 'category_id' 列。")
            return None

        print(f"加载成功: {len(df)} 行数据，{df['category_id'].nunique()} 个类别。")
        return df
    except FileNotFoundError:
        print(f"错误: 未找到聚类结果文件 '{csv_path}'。")
        print("请确保此脚本与您的聚类结果文件在同一目录中。")
        return None
    except Exception as e:
        print(f"加载文件时出错: {e}")
        return None


# --- 4. 整合、筛选并保存 ---
def summarize_and_save(df, key_features_list, output_path):
    print("正在按 'category_id' 整合数据 (计算均值)...")

    cluster_centers_all_features = df.groupby('category_id').mean(numeric_only=True)

    print(f"正在从 {len(cluster_centers_all_features.columns)} 个总特征中筛选 {len(key_features_list)} 个关键特征...")

    existing_features = [col for col in key_features_list if col in cluster_centers_all_features.columns]
    missing_features = [col for col in key_features_list if col not in cluster_centers_all_features.columns]

    if missing_features:
        print(f"警告: {len(missing_features)} 个关键特征未在文件中找到，已跳过。")
        print(f"  -> 示例: {missing_features[:5]}")

    if not existing_features:
        print("错误：特征列表 中的所有特征（带 _mean/_std/etc 后缀）都无法在聚类文件中找到。")
        return

    # 筛选
    llm_input_df = cluster_centers_all_features[existing_features]

    # 转置 DataFrame，以便于 LLM 读取
    llm_input_df_transposed = llm_input_df.transpose()

    try:
        # 保存为 CSV
        llm_input_df.to_csv(output_path, encoding='utf-8-sig', float_format='%.2f')
        print(f"\n--- 成功! ---")
        print(f"已将 {len(llm_input_df)} 个类别的“完整画像”保存到: {output_path}")

        print("\n--- 每个类别的“完整画像”（用于 LLM 分析） ---")
        pd.set_option('display.max_rows', 500)
        print(llm_input_df_transposed.round(2))

    except Exception as e:
        print(f"保存文件时出错: {e}")


# --- 主程序执行 ---
if __name__ == "__main__":

    key_features_for_llm = load_features_from_json(FEATURES_JSON_FILE)

    if key_features_for_llm:
        clustered_df = load_clustered_data(CLUSTERED_DATA_FILE)

        if clustered_df is not None:

            summarize_and_save(clustered_df, key_features_for_llm, OUTPUT_FILE)
