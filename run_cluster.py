# ==========================================================
# 脚本 3 (最终版 - 已修复 NameError): run_global_search_with_manual_decision.py
# (此脚本允许您在自动搜索后，手动选择最佳的 {算法, N, K} 组合)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 进度条

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture  # (来自 test3.py)
from sklearn.metrics import silhouette_score
import sys
import time
import os

# (确保您已安装: pip install pyarrow scikit-learn seaborn)

# --- 全局设置：确保绘图能正确显示中文和负号 ---
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)


# --- 步骤 1: N-K-算法 联合搜索 (基于 test2.py 和 test3.py) ---
# --- 【已修复】添加 'detail_plot_dir' 参数 ---
def step1_global_search(features_df, n_range, k_range, detail_plot_dir):
    """
    (函数顶部的文档字符串保持不变)
    ...
    """
    print("\n--- 步骤 1: 开始联合搜索最佳算法, N (PCA) 和 K (KMeans/GMM)... ---")

    # --- 1. 标准化 (只做一次) ---
    feature_cols = [col for col in features_df.columns if col != 'window_start_time']
    X = features_df[feature_cols].values

    if X.shape[0] == 0:
        print("错误：特征文件为空，无法进行聚类。")
        return None, None, None, None, None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- 2. 完整 PCA 分析 (只做一次) ---
    print("正在拟合完整PCA以分析方差...")
    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(X_scaled)

    # (绘图逻辑来自 test2.py)
    plt.figure(figsize=(10, 6))

    # --- 【关键】: 我们在这里获取方差数组 ---
    explained_variance = np.cumsum(pca_full.explained_variance_ratio_)
    # --- 【结束】 ---

    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('主成分数量')
    plt.ylabel('累计解释方差比例')
    plt.title('PCA 累计解释方差')
    plt.grid(True)
    plt.savefig('pca_explained_variance_plot.png', dpi=300)
    print("PCA 累计解释方差图已保存为: pca_explained_variance_plot.png")

    X_pca_full = pca_full.transform(X_scaled)

    # (抽样逻辑来自 test3.py)
    if len(X_pca_full) > 10000:
        print(f"数据量 ({len(X_pca_full)}) > 10000, 剪影系数将使用 10000 样本进行估算。")
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(X_pca_full), 10000, replace=False)
        X_sample_full = X_pca_full[sample_indices]
    else:
        print(f"数据量 ({len(X_pca_full)}) <= 10000, 将在完整数据上计算剪影系数。")
        X_sample_full = X_pca_full
        sample_indices = None

    os.makedirs(detail_plot_dir, exist_ok=True)
    print(f"K vs Score 的详细图表将保存在: ./{detail_plot_dir}/")

    # --- 3. 嵌套循环搜索 ---
    kmeans_search_results = []  # 存储 (n, best_k_for_n, best_score_for_n)
    gmm_search_results = []

    outer_pbar = tqdm(n_range, desc="搜索 N (PCA 维度)")
    for n in outer_pbar:
        # ( ... 内部循环 ... )
        # ( ... 内部循环的 K-loop 保持不变 ... )
        current_X_pca = X_pca_full[:, :n]
        current_X_sample = X_sample_full[:, :n]
        k_scores_kmeans = []
        k_scores_gmm = []
        best_score_k_for_this_n = -1;
        best_k_k = 0
        best_score_g_for_this_n = -1;
        best_k_g = 0
        for k in k_range:
            score_k = -1
            score_g = -1
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(current_X_pca)
                if sample_indices is not None:
                    labels_k = kmeans.predict(current_X_sample)
                    score_k = silhouette_score(current_X_sample, labels_k)
                else:
                    score_k = silhouette_score(current_X_pca, kmeans.labels_)
                if score_k > best_score_k_for_this_n:
                    best_score_k_for_this_n = score_k
                    best_k_k = k
            except Exception as e_k:
                pass
            k_scores_kmeans.append(score_k)
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, n_init=5, reg_covar=1e-6)
                gmm.fit(current_X_pca)
                if sample_indices is not None:
                    labels_g = gmm.predict(current_X_sample)
                    score_g = silhouette_score(current_X_sample, labels_g)
                else:
                    labels_g = gmm.predict(current_X_pca)
                    score_g = silhouette_score(current_X_pca, labels_g)
                if score_g > best_score_g_for_this_n:
                    best_score_g_for_this_n = score_g
                    best_k_g = k
            except Exception as e_g:
                pass
            k_scores_gmm.append(score_g)
            # ( ... K-loop 结束 ... )
        kmeans_search_results.append((n, best_k_k, best_score_k_for_this_n))
        gmm_search_results.append((n, best_k_g, best_score_g_for_this_n))

        # --- 4. 【新】为这个N绘制并保存 K vs Score 详细图 ---
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, k_scores_kmeans, marker='o', label='KMeans')
        plt.plot(k_range, k_scores_gmm, marker='s', label='GMM')
        plt.title(f'K vs. 剪影系数 (当 PCA 维度 N = {n})')
        plt.xlabel(f'聚类数量 (K) (N={n}时, KMeans最佳K={best_k_k}, GMM最佳K={best_k_g})')
        plt.ylabel('平均剪影系数')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(detail_plot_dir, f'k_vs_score_for_N_{n}.png'), dpi=150)
        plt.close()  # 关闭图表以节省内存

    print("--- 搜索完成 ---")

    # --- 5. 分析并推荐结果 ---
    kmeans_best = max(kmeans_search_results, key=lambda item: item[2])
    gmm_best = max(gmm_search_results, key=lambda item: item[2])

    # --- 【修改】 ---
    # (因为 N 是从 1 开始的，所以 N=5 对应索引 4)
    kmeans_best_n = kmeans_best[0]
    kmeans_variance = explained_variance[kmeans_best_n - 1]
    print(f"\n--- 最佳 KMeans 结果 (自动推荐) ---")
    print(f"  -> N={kmeans_best_n}, K={kmeans_best[1]}, 剪影系数={kmeans_best[2]:.4f}")
    print(f"  -> N={kmeans_best_n} 时, 保留了 {kmeans_variance:.2%} 的原始方差 (信息)。")
    # --- 【修改结束】 ---

    # --- 【修改】 ---
    gmm_best_n = gmm_best[0]
    gmm_variance = explained_variance[gmm_best_n - 1]
    print(f"\n--- 最佳 GMM 结果 (自动推荐) ---")
    print(f"  -> N={gmm_best_n}, K={gmm_best[1]}, 剪影系数={gmm_best[2]:.4f}")
    print(f"  -> N={gmm_best_n} 时, 保留了 {gmm_variance:.2%} 的原始方差 (信息)。")
    # --- 【修改结束】 ---

    if kmeans_best[2] > gmm_best[2]:
        recommended_algo = 'kmeans'
        recommended_n = kmeans_best[0]
        recommended_k = kmeans_best[1]
        recommended_score = kmeans_best[2]
    else:
        recommended_algo = 'gmm'
        recommended_n = gmm_best[0]
        recommended_k = gmm_best[1]
        recommended_score = gmm_best[2]

    # --- 【修改】 ---
    global_variance = explained_variance[recommended_n - 1]
    print(f"\n--- 全局冠军 (自动推荐) ---")
    print(f"  -> 算法: {recommended_algo.upper()}")
    print(f"  -> 最佳 PCA 维度 (N): {recommended_n}")
    print(f"  -> 最佳聚类数 (K): {recommended_k}")
    print(f"  -> 最高剪影系数: {recommended_score:.4f}")
    print(f"  -> {recommended_n} 维保留了 {global_variance:.2%} 的原始方差 (信息)。")
    # --- 【修改结束】 ---

    # --- 绘制 N vs Score 总览图 (来自 test2.py) ---
    results_df_k = pd.DataFrame(kmeans_search_results, columns=['N_Components', 'Best_K', 'Silhouette_Score'])
    results_df_g = pd.DataFrame(gmm_search_results, columns=['N_Components', 'Best_K_GMM', 'Silhouette_Score_GMM'])

    plt.figure(figsize=(12, 7))
    plt.plot(results_df_k['N_Components'], results_df_k['Silhouette_Score'], marker='o',
             label='KMeans (在N上的最佳K得分)')
    plt.plot(results_df_g['N_Components'], results_df_g['Silhouette_Score_GMM'], marker='s',
             label='GMM (在N上的最佳K得分)')
    plt.xlabel('PCA 主成分数量 (N)')
    plt.ylabel('该 N 对应的最佳剪影系数')
    plt.title('全局搜索: N (PCA维度) vs 最佳剪影系数')
    plt.legend()
    plt.grid(True)
    plt.savefig('global_search_n_vs_score_SUMMARY.png', dpi=300)
    print("全局搜索总览图已保存为: global_search_n_vs_score_SUMMARY.png")

    # --- 6. 返回推荐值和 *完整* 的 X_scaled / pca_full ---
    return X_scaled, pca_full, recommended_algo, recommended_n, recommended_k


# --- 步骤 2: 【新】应用用户选择的PCA ---
def step2_apply_final_pca(X_scaled, pca_full, final_n):
    """
    一个简单的新函数，用于从完整的PCA结果中
    切片出用户最终选择的N维数据。
    """
    print(f"\n--- 步骤 2: 正在应用您选择的 N={final_n} 维度 ---")
    # 从完整的转换结果中“切片”出前 N 维
    X_pca_final = pca_full.transform(X_scaled)[:, :final_n]
    print(f"已生成最终的 {X_pca_final.shape[0]} x {X_pca_final.shape[1]} 维特征矩阵。")
    return X_pca_final


# --- 步骤 3: 聚类与可视化 (来自 test3.py) ---
def step3_cluster_and_visualize(X_pca, algo_name, optimal_k):
    """
    (原 test3.py 中的 step4_cluster_and_visualize)
    1. 使用“冠军”算法和K值执行最终聚类。
    2. 使用 PC1 和 PC2 绘制聚类结果散点图。
    """
    print(f"\n--- 步骤 3: 正在使用 {algo_name.upper()} (K={optimal_k}) 执行最终聚类... ---")

    if algo_name == 'kmeans':
        model = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    elif algo_name == 'gmm':
        model = GaussianMixture(n_components=optimal_k, random_state=42, n_init=5, reg_covar=1e-6)

    start_time = time.time()
    cluster_labels = model.fit_predict(X_pca)
    print(f"最终聚类耗时: {time.time() - start_time:.2f} 秒")

    print("正在生成可视化图表...")

    # (绘图逻辑来自 test3.py)
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'cluster': cluster_labels.astype(str)
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=plot_df,
        x='PC1',
        y='PC2',
        hue='cluster',
        palette='viridis',
        s=50,
        alpha=0.7
    )
    plt.title(f'最终聚类结果 (PC1 vs PC2)\n算法: {algo_name.upper()}, N={X_pca.shape[1]}, K={optimal_k}')
    plt.xlabel('主成分 1 (PC1)')
    plt.ylabel('主成分 2 (PC2)')
    plt.legend(title='聚类类别', loc='best')
    plt.grid(True)
    # (文件名包含了N和K，以便区分)
    plt.savefig(f'cluster_visualization_FINAL_{algo_name}_n{X_pca.shape[1]}_k{optimal_k}.png', dpi=300)
    print(f"聚类可视化图表已保存。")

    return cluster_labels


# --- 步骤 4: 反向映射与保存 (来自 test3.py) ---
def step4_save_results(features_df, cluster_labels, output_filename):
    """
    (原 test3.py 中的 step5_save_results)
    1. 将聚类标签 (category_id) 添加回 "未做PCA" 的特征数据上。
    2. 保存到新的 CSV 文件。
    """
    print(f"\n--- 步骤 4: 正在反向映射并保存最终结果... ---")

    final_df = features_df.copy()
    final_df['category_id'] = cluster_labels

    cols = list(final_df.columns)
    cols.remove('category_id')
    cols.remove('window_start_time')
    final_cols_order = ['category_id', 'window_start_time'] + cols
    final_df = final_df[final_cols_order]

    try:
        final_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"--- 流程完毕 ---")
        print(f"最终结果已成功保存到: {output_filename}")
        print(f"  共 {len(final_df)} 个驾驶窗口")
        print(f"  共 {final_df['category_id'].nunique()} 个类别")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


# --- 主程序执行 (脚本 3) ---
if __name__ == '__main__':

    FEATURES_FILEPATH = 'driving_features.parquet'
    FINAL_OUTPUT_FILEPATH = 'clustered_driving_segments_MANUAL_CHOICE.csv'

    # (基于您 {N=11, K=14} 的发现，我们扩大 K 的搜索范围)
    N_COMPONENTS_RANGE = range(2, 16)  # 搜索 8 到 15 维
    K_RANGE = range(2, 18)  # 搜索 2 到 17 簇

    # --- 【已修复】在 main 作用域中定义目录名 ---
    DETAIL_PLOT_DIR = "k_vs_score_plots"
    # --- 【修复结束】 ---

    # --- 1. 加载特征文件 ---
    try:
        print(f"--- 开始全局优化搜索 (KMeans vs GMM) 流程 ---")
        print(f"正在从 {FEATURES_FILEPATH} 加载特征...")
        features_df = pd.read_parquet(FEATURES_FILEPATH)
        print(f"加载成功: {len(features_df)} 个窗口，{len(features_df.columns)} 个特征。")
    except FileNotFoundError:
        print(f"错误: 未找到特征文件 '{FEATURES_FILEPATH}'。")
        print("请先运行 'run_feature_engineering.py' 脚本。")
        sys.exit(1)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        sys.exit(1)

    if features_df is not None:

        # --- 2. 步骤 1: 运行全局搜索 ---
        # --- 【已修复】将 'DETAIL_PLOT_DIR' 传递给函数 ---
        X_scaled, pca_full, rec_algo, rec_n, rec_k = step1_global_search(
            features_df,
            n_range=N_COMPONENTS_RANGE,
            k_range=K_RANGE,
            detail_plot_dir=DETAIL_PLOT_DIR
        )

        # --- 3. 【新】用户决策步骤 ---
        print("\n" + "=" * 50)
        print("--- 请您进行最终决策 ---")
        print("脚本已完成自动搜索，并提供了“自动推荐”。")
        print(f"请查看 'global_search_n_vs_score_SUMMARY.png' (总览图)")
        # --- 【已修复】使用在 main 作用域中定义的变量 ---
        print(f"以及 './{DETAIL_PLOT_DIR}/' 文件夹中的详细图表 (例如 'k_vs_score_for_N_{rec_n}.png')。")
        print("\n--- 简约原则提醒 ---")
        print(f"自动推荐的 K={rec_k}。如果 K={rec_k - 1} 或 K={rec_k - 2} 的分数几乎一样高，")
        print("您可能应该选择那个 K 值更小的模型，因为它更容易解释。")
        print("=" * 50 + "\n")

        # 接收用户输入 (如果用户直接按回车，则使用推荐值)
        final_algo = input(f"请输入您最终选择的算法 (kmeans/gmm) [默认为: {rec_algo}]: ").strip().lower() or rec_algo

        try:
            final_n_str = input(f"请输入您最终选择的 N 维度 [默认为: {rec_n}]: ") or str(rec_n)
            final_n = int(final_n_str)
        except ValueError:
            print(f"输入无效，将使用默认 N={rec_n}")
            final_n = rec_n

        try:
            final_k_str = input(f"请输入您最终选择的 K 值 [默认为: {rec_k}]: ") or str(rec_k)
            final_k = int(final_k_str)
        except ValueError:
            print(f"输入无效，将使用默认 K={rec_k}")
            final_k = rec_k

        print(f"\n--- 您的最终选择: 算法={final_algo}, N={final_n}, K={final_k} ---")

        # --- 4. 步骤 2: 应用用户选择的 N ---
        # (从完整的 X_scaled 和 pca_full 中切片出用户要的 N 维数据)
        X_pca_final = pca_full.transform(X_scaled)[:, :final_n]

        # --- 5. 步骤 3: 聚类与可视化 ---
        cluster_labels = step3_cluster_and_visualize(X_pca_final, final_algo, final_k)

        # --- 6. 步骤 4: 反向映射与保存 ---
        step4_save_results(features_df, cluster_labels, FINAL_OUTPUT_FILEPATH)

    else:
        print("特征文件加载失败，程序终止。")