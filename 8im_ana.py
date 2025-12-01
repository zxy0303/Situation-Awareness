import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def run_importance_analysis():
    print("\n--- 开始运行：特征重要性深度分析 (修复版) ---")

    # 1. 加载数据
    # 请确保文件路径正确
    csv_path = 'labeled_source_data_with_situations.csv'
    json_path = 'selected_features_structured.json'

    try:
        df = pd.read_csv(csv_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 建立特征映射
    situation_feature_map = {}
    for category in structured_data:
        for sub_cat in category['sub_categories']:
            situation_feature_map[sub_cat['sub_category_name']] = sub_cat['features']

    # 识别态势列
    situation_cols_csv = df.columns[-8:].tolist()
    col_to_json_key = {}
    for col in situation_cols_csv:
        for key in situation_feature_map.keys():
            if col.endswith(key):
                col_to_json_key[col] = key
                break

    # 2. 分析函数
    def analyze_single_situation(sit_col, feature_list):
        print(f"\n>>> 正在分析态势: {sit_col}")

        valid_features = [f for f in feature_list if f in df.columns]
        if not valid_features:
            print("  跳过：无有效特征")
            return

        X = df[valid_features].copy()
        y = df[sit_col].fillna('Unknown').astype(str)

        if y.nunique() <= 1:
            print("  跳过：态势数据无变化 (常量)")
            return

        # --- 修复点 1: 数据清洗与空列处理 ---
        # 先转换为数值型
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # 【关键修复】：在 Imputer 之前，先删除全为 NaN 的列
        # 这样 X.columns 就和 Imputer 处理后的列数一致了
        X.dropna(axis=1, how='all', inplace=True)

        if X.shape[1] == 0:
            print("  跳过：所有特征均为全空 (NaN)")
            return

        # 填充缺失值
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        # 再次去除常量特征 (方差为0)
        X_imputed = X_imputed.loc[:, X_imputed.std() > 1e-6]
        final_features = X_imputed.columns.tolist()

        if not final_features:
            print("  跳过：特征均为常量")
            return

        # 标签编码
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # --- A. 随机森林特征重要性 ---
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_imputed, y_encoded)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("  [Top 5 重要特征]")
        # 确保索引不越界
        n_top = min(5, len(indices))
        top_feature = final_features[indices[0]]
        for i in range(n_top):
            print(f"    {i + 1}. {final_features[indices[i]]}: {importances[indices[i]]:.4f}")

        # 绘制重要性条形图
        plt.figure(figsize=(10, 6))
        pd.Series(importances, index=final_features).nlargest(10).plot(kind='barh').invert_yaxis()
        plt.title(f'Feature Importance: {sit_col}')
        plt.tight_layout()
        plt.savefig(f'importance_{sit_col}.png')
        print(f"  图表已保存: importance_{sit_col}.png")

        # --- B. 箱线图 (Top 1 特征分布) ---
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=y, y=X_imputed[top_feature])
        plt.title(f'{top_feature} vs {sit_col}')
        plt.tight_layout()
        plt.savefig(f'boxplot_{sit_col}.png')
        print(f"  图表已保存: boxplot_{sit_col}.png")

        # --- C. 决策树规则 (阈值提取) ---
        dt = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt.fit(X_imputed, y_encoded)
        rules = export_text(dt, feature_names=final_features)
        print(f"  [关键规则建议]:\n{rules}")

    # 3. 执行循环
    for sit_col in situation_cols_csv:
        json_key = col_to_json_key.get(sit_col)
        if json_key:
            analyze_single_situation(sit_col, situation_feature_map[json_key])


if __name__ == "__main__":
    # 设置绘图风格
    sns.set(style="whitegrid")
    # 解决中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    run_importance_analysis()