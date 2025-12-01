import pandas as pd
import json
import sys

# --- 1. 定义文件路径 (请根据需要修改) ---

# 源数据文件 (来自 data_processing.py 的输出)
SOURCE_DATA_FILE = '2025-08-23_processed_Q10-Q90.csv'

# "桥梁"文件 (来自 run_cluster.py 的输出)
SEGMENTS_FILE = 'clustered_driving_segments_MANUAL_CHOICE.csv'

# LLM 标签文件
JSON_LABELS_FILE = 'cluster_labels_output_G.json'

# 最终输出文件
OUTPUT_FILE = 'labeled_source_data_with_situations.csv'

WINDOW_SIZE_SECONDS = 60
# --- 2. 【已修复】解析 JSON 态势文件 ---
def load_and_parse_json_labels(json_path):
    """
    加载 JSON 文件 (长格式)，并将其转换为宽格式的 DataFrame。

    【修复】:
    使用 pivot_table 和 aggfunc='first' 来代替 pivot，
    以处理 (Cluster_ID, Subdimension) 组合在 JSON 中重复出现的问题。
    """
    print(f"正在加载JSON态势文件: {json_path}")

    try:
        df_long = pd.read_json(json_path, encoding='utf-8-sig')
    except FileNotFoundError:
        print(f"错误: JSON文件 {json_path} 未找到。")
        sys.exit(1)
    except ValueError as e:
        print(f"错误: JSON文件 {json_path} 格式错误。错误信息: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"读取JSON时发生未知错误: {e}")
        sys.exit(1)

    # 检查必需的列
    required_cols = ['Cluster_ID', 'Dimension', 'Subdimension', 'Assigned_Label']
    if not all(col in df_long.columns for col in required_cols):
        print(f"错误: JSON 文件缺少必需的列。")
        print(f"需要: {required_cols}")
        print(f"找到: {df_long.columns.tolist()}")
        sys.exit(1)

    print("JSON 加载成功，正在进行数据透视 (pivot)...")

    # 1. 创建组合列名 (例如 "Environment_weather&thermal")
    df_long['situation_col_name'] = df_long['Dimension'] + '_' + df_long['Subdimension']

    # 2. 数据透视：将长格式转为宽格式
    try:
        # --- 【修复】---
        # 使用 pivot_table 代替 pivot，并添加 aggfunc='first'
        # 这将自动处理重复条目，只保留第一个出现的标签
        labels_df = df_long.pivot_table(
            index='Cluster_ID',
            columns='situation_col_name',
            values='Assigned_Label',
            aggfunc='first'  # <-- 关键修复：处理重复项
        )
        # --- 【修复结束】---

    except Exception as e:
        # (保留此处的 try/except 以防万一，尽管 pivot_table 应该已解决问题)
        print(f"错误：在数据透视时出错: {e}")
        sys.exit(1)

    # 3. 获取所有态势列的列表
    situation_columns_found = labels_df.columns.tolist()

    print(f"成功解析 {len(labels_df)} 个聚类的态势。")
    print(f"共找到 {len(situation_columns_found)} 个态势子维度 (已排除 'Reason')。")

    example_cols = situation_columns_found[:4]
    print(f"  -> 示例列名: {example_cols}")

    # 重置索引，使 Cluster_ID 成为一个列（如果需要的话），但 merge 时用索引更好
    # labels_df = labels_df.reset_index()

    return labels_df, situation_columns_found


# --- 3. 主执行函数 (无需修改) ---
def main():
    # --- 步骤 1: 加载并处理JSON态势标签 ---
    # (此函数已被修复)
    labels_df, situation_columns = load_and_parse_json_labels(JSON_LABELS_FILE)

    # --- 步骤 2: 加载聚类分段文件 (category_id <-> window_start_time) ---
    print(f"正在加载分段文件: {SEGMENTS_FILE}")
    try:
        segments_df = pd.read_csv(
            SEGMENTS_FILE,
            usecols=['window_start_time', 'category_id'],
            parse_dates=['window_start_time']
        )
    except FileNotFoundError:
        print(f"错误: 分段文件 {SEGMENTS_FILE} 未找到。")
        return
    except ValueError:
        print(f"错误: {SEGMENTS_FILE} 中未找到 'window_start_time' 或 'category_id' 列。")
        return
    print(f"正在根据 {WINDOW_SIZE_SECONDS}s 窗口大小计算结束时间...")
    segments_df['window_end_time'] = segments_df['window_start_time'] + pd.Timedelta(seconds=WINDOW_SIZE_SECONDS)

    # --- 步骤 3: 将态势标签合并到分段中 ---
    segment_labels_df = segments_df.merge(
        labels_df,
        left_on='category_id',
        right_index=True,  # (labels_df 的索引是 Cluster_ID)
        how='left'
    )

    segment_labels_df = segment_labels_df.sort_values('window_end_time')
    print("已合并态势标签到分段时间。")

    # --- 步骤 4: 加载源数据文件 ---
    print(f"正在加载源数据文件: {SOURCE_DATA_FILE}")
    try:
        original_df = pd.read_csv(
            SOURCE_DATA_FILE,
            parse_dates=['collectTime_dt']
        )
    except FileNotFoundError:
        print(f"错误: 源数据文件 {SOURCE_DATA_FILE} 未找到。")
        return
    except ValueError:
        print(f"错误: {SOURCE_DATA_FILE} 中未找到 'collectTime_dt' 列。")
        return

    original_df = original_df.sort_values('collectTime_dt')
    print(f"源数据加载成功，总计 {len(original_df)} 行。")

    # --- 步骤 5: 将分段态势标注回源文件 (核心步骤) ---
    print("正在将态势标注合并回源文件 (使用 merge_asof)...")

    columns_to_merge = ['window_end_time'] + situation_columns

    labeled_original_df = pd.merge_asof(
        original_df,
        segment_labels_df[columns_to_merge],
        left_on='collectTime_dt',
        right_on='window_end_time',
        direction='forward'
    )

    labeled_original_df = labeled_original_df.drop(columns=['window_end_time'])

    # --- 步骤 6: 保存结果 ---
    try:
        labeled_original_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n--- 成功! ---")
        print(f"已将标注回的源文件保存到: {OUTPUT_FILE}")
        print(f"最终文件形状: {labeled_original_df.shape}")
        print(f"新增的态势列 (共 {len(situation_columns)} 个): {situation_columns}")
    except Exception as e:
        print(f"\n保存文件失败: {e}")


if __name__ == "__main__":
    main()