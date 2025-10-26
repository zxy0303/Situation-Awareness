import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm 
import os  

##此文件用来训练聚类，并固定聚类数量，输出聚类结果后程序暂停，将聚类结果利用GPT标注Status_code后返回程序运行，程序根据标签将同类中的样本数据统一打标签
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False



def feature_engineering_sliding_window(filepath, trip_split_minutes=10, window_seconds=60, step_seconds=10):

    print("开始加载数据...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, low_memory=False, encoding='gbk')

    print("数据加载完毕，开始进行特征工程...")


    try:
        df['collectTime'] = df['collectTime'].astype(np.int64)
    except ValueError as e:
        print(f"错误：receiveTime 列转换失败: {e}。请检查数据中是否有非数字值。")
        return pd.DataFrame()


    print("正在转换时间戳格式 (创建辅助列)...")
    df['collectTime_dt'] = pd.to_datetime(df['collectTime'], unit='ms')

    df = df.sort_values('collectTime_dt').reset_index(drop=True)
    time_diff = df['collectTime_dt'].diff().dt.total_seconds() / 60
    df['trip_id'] = (time_diff > trip_split_minutes).cumsum()
    window_features = []

    for trip_id, trip_data in df.groupby('trip_id'):

        if trip_data['vehicleSpeed'].max() == 0:
            continue

        trip_data = trip_data.reset_index(drop=True)
        start_time = trip_data['collectTime_dt'].min()
        end_time = trip_data['collectTime_dt'].max()
        print(f"\n正在处理行程 {trip_id} (时长: {end_time - start_time})...")
        current_time = start_time
        pbar_total_seconds = (end_time - start_time).total_seconds()
        pbar = tqdm(total=max(pbar_total_seconds, step_seconds))

        while current_time + pd.Timedelta(seconds=window_seconds) <= end_time:
            window_start = current_time
            window_end = current_time + pd.Timedelta(seconds=window_seconds)
            window_df = trip_data[(trip_data['collectTime_dt'] >= window_start) &
                                  (trip_data['collectTime_dt'] < window_end)]

            if len(window_df) > 5:
                mileage_covered = window_df['totalMileage'].iloc[-1] - window_df['totalMileage'].iloc[0]
                if mileage_covered < 0: mileage_covered = 0
                actual_start_time_int = window_df['collectTime'].iloc[0]


                window_features.append({
                    'start_time_int': actual_start_time_int,

                    'avg_speed': window_df['vehicleSpeed'].mean(),
                    'max_speed': window_df['vehicleSpeed'].max(),
                    'std_speed': window_df['vehicleSpeed'].std(),
                    'mileage_covered': mileage_covered,
                    'avg_actGearSt': window_df['actGearSt'].mean(),
                    'max_actGearSt': window_df['actGearSt'].max(),
                    'std_actGearSt': window_df['actGearSt'].std(),
                    'avg_SOC': window_df['battSocAct'].mean(),
                    'avg_outsidetem': window_df['outsideAmbTemp'].mean(),
                    'avg_accmode': window_df['accMode'].mean(),
                    'avg_driveMode': window_df['driveMode'].mean(),
                    'avg_rlTyreTemp':window_df['rlTyreTemp'].mean(),
                    'max_rlTyreTemp':window_df['rlTyreTemp'].max(),
                    'min_rlTyreTemp':window_df['rlTyreTemp'].min(),
                    'avg_rlTyrePressure':window_df['rlTyrePressure'].mean(),
                    'max_rlTyrePressure':window_df['rlTyrePressure'].max(),
                    'min_rlTyrePressure':window_df['rlTyrePressure'].min(),
                    'avg_rrTyreTemp': window_df['rrTyreTemp'].mean(),
                    'max_rrTyreTemp': window_df['rrTyreTemp'].max(),
                    'min_rrTyreTemp': window_df['rrTyreTemp'].min(),
                    'avg_rrTyrePressure': window_df['rrTyrePressure'].mean(),
                    'max_rrTyrePressure': window_df['rrTyrePressure'].max(),
                    'min_rrTyrePressure': window_df['rrTyrePressure'].min(),
                    'avg_flTyreTemp': window_df['flTyreTemp'].mean(),
                    'max_flTyreTemp': window_df['flTyreTemp'].max(),
                    'min_flTyreTemp': window_df['flTyreTemp'].min(),
                    'avg_flTyrePressure': window_df['flTyrePressure'].mean(),
                    'max_flTyrePressure': window_df['flTyrePressure'].max(),
                    'min_flTyrePressure': window_df['flTyrePressure'].min(),
                    'avg_frTyreTemp': window_df['frTyreTemp'].mean(),
                    'max_frTyreTemp': window_df['frTyreTemp'].max(),
                    'min_frTyreTemp': window_df['frTyreTemp'].min(),
                    'avg_frTyrePressure': window_df['frTyrePressure'].mean(),
                    'max_frTyrePressure': window_df['frTyrePressure'].max(),
                    'min_frTyrePressure': window_df['frTyrePressure'].min(),
                    'avg_accpedalAct':window_df['accPedalActualPosition'].mean(),
                    'max_accpedalAct': window_df['accPedalActualPosition'].max(),
                    'std_accpedalAct': window_df['accPedalActualPosition'].std(),
                    'avg_longAcceleration': window_df['longAcceleration'].mean(),
                    'std_longAcceleration': window_df['longAcceleration'].std(),
                    'avg_lateralAcceleration': window_df['lateralAcceleration'].mean(),
                    'std_lateralAcceleration': window_df['lateralAcceleration'].std(),
                    'avg_actHSRelHum': window_df['actHSRelHum'].mean(),
                })

            current_time += pd.Timedelta(seconds=step_seconds)
            pbar.update(min(step_seconds, pbar.total - pbar.n))
        pbar.close()

    features_df = pd.DataFrame(window_features).dropna()
    print(f"\n特征工程完成，共从所有行程中提取出 {len(features_df)} 个有效的时间窗口。")
    try:
        print(f"正在将特征文件保存到:")
        features_df.to_csv('./result.csv', index=False, encoding='utf-8-sig')
        print("文件保存成功！ (./result.csv)")
    except Exception as e:
        print(f"文件保存失败: {e}")
    return features_df

def train_clustering_model(features_df):
    if features_df is None or features_df.empty:
        print("特征数据为空，无法进行聚类。")
        return None, None, None
    print("\n开始训练聚类模型...")
    feature_cols = [col for col in features_df.columns if col not in ['start_time_int']]
    X = features_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sse = []
    k_range = range(2, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('聚类数量 (K)')
    plt.ylabel('SSE (簇内误差平方和)')
    plt.title('肘部法则确定最佳K值')
    plt.grid(True)
    plt.savefig('elbow_plot.png', dpi=300)
    print("肘部法则图已保存为 elbow_plot.png")
    plt.close()

    optimal_k = 7
    print(f"\n已自动设定 K = {optimal_k} (如需更改请修改代码)")

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    features_df['cluster'] = kmeans.fit_predict(X_scaled)
    print(f"模型训练完成，已将数据聚为 {optimal_k} 类。")
    # (features_df 现在包含 'start_time_int' 和 'cluster' 列)
    return features_df, feature_cols, scaler, kmeans

def analyze_and_visualize(result_df, feature_cols, scaler, kmeans):
    if result_df is None:
        return None
    print("\n--- 聚类结果分析 ---")

    k = result_df['cluster'].nunique()
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers_real = pd.DataFrame(scaler.inverse_transform(cluster_centers_scaled), columns=feature_cols,
                                        index=range(k))

    cluster_centers_real.index.name = 'cluster_id'

    print("每个类别的窗口数量:")
    print(result_df['cluster'].value_counts().sort_index())
    print("\n每个类别的特征均值 (真实类别数值):")
    print(cluster_centers_real.round(2))

    centers_file = f'cluster_centers_analysis_{k}.csv'
    cluster_centers_real.round(2).to_csv(centers_file, encoding='utf-8-sig')
    print(f"\n聚类中心特征均值已保存为 {centers_file}")

    print("\n--- 交互式状态标注 ---")
    print(f"1. 脚本已暂停。请在程序目录中找到并打开: {centers_file}")
    print(f"2. 在该文件的最后一列，*新添加*一列，列名为 'Status_Code'")
    print(f"3. 为每一行(代表一个聚类)填写您定义的标签。")
    print(f"4. !! 重要：请确保列名完全正确 ('Status_Code')，然后保存并关闭该CSV文件。 !!")

    try:
        input("\n...完成后，请按 Enter 键继续...")
    except EOFError:
        print("\n输入被中断，退出。")
        return None

    print("正在读取已标注的文件，并合并到总数据中...")
    status_map = {}
    try:
        labeled_centers_df = pd.read_csv(centers_file, index_col=0, encoding='utf-8-sig')

        if 'Status_Code' not in labeled_centers_df.columns:
            print(f"错误：在 {centers_file} 中未找到名为 'Status_Code' 的列。")
            print("请确保您已正确添加该列并保存了文件。分析中止。")
            return None

        status_map = labeled_centers_df['Status_Code'].to_dict()
        print(f"成功读取到以下状态映射: {status_map}")
        result_df['Status_Code'] = result_df['cluster'].map(status_map)
        output_file = f'all_samples_with_status_labels_k{k}.csv'
        result_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n--- 成功 ---")
        print(f"已将 'Status_Code' 标签添加到所有 {len(result_df)} 个样本窗口中。")
        print(f"完整的、已标注的样本数据已保存到: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {centers_file}。")
        return None
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return None

    print("\n正在生成雷达图...")
    labels = np.array(feature_cols)
    n_labels = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)

    for i, stats in enumerate(cluster_centers_scaled):
        count = result_df['cluster'].value_counts().get(i, 0)
        label_name = status_map.get(i, f'类别 {i}')

        stats = np.concatenate((stats, [stats[0]]))
        ax.plot(angles, stats, label=f"{label_name} (ID: {i}, 数量: {count})")
        ax.fill(angles, stats, alpha=0.1)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title("不同驾驶态势聚类结果雷达图 (标准化后)")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    radar_file = f'radar_chart_{k}_labeled.png'
    plt.savefig(radar_file, dpi=300, bbox_inches='tight')
    print(f"雷达图已保存为 {radar_file}")
    plt.close()

    print("聚类分析和标注完成。")
    return result_df



def map_labels_to_original_file(original_filepath, labeled_windows_df):
    print("\n--- 正在将工况标签映射回原始VHR文件 ---")
    if 'start_time_int' not in labeled_windows_df.columns or 'Status_Code' not in labeled_windows_df.columns:
        print("错误：标注的窗口数据中缺少 'start_time_int' 或 'Status_Code' 列。")
        return

    labeled_windows_df['start_time_int'] = labeled_windows_df['start_time_int'].astype(np.int64)
    lookup_df = labeled_windows_df[['start_time_int', 'Status_Code']].drop_duplicates()
    print(f"已加载 {len(lookup_df)} 个唯一的窗口标签。")
    print(f"正在加载原始文件: {original_filepath} ...")
    try:
        original_df = pd.read_csv(original_filepath, low_memory=False)
    except UnicodeDecodeError:
        original_df = pd.read_csv(original_filepath, low_memory=False, encoding='gbk')
    try:
        original_df['collectTime'] = original_df['collectTime'].astype(np.int64)
    except ValueError as e:
        print(f"错误：原始文件 {original_filepath} 的 collectTime 列转换失败: {e}。")
        return
    print("正在合并标签到原始数据...")
    merged_df = pd.merge(
        original_df,
        lookup_df,
        left_on='collectTime',   
        right_on='start_time_int',
        how='left'
    )

    if 'start_time_int' in merged_df.columns:
        merged_df = merged_df.drop(columns=['start_time_int'])

    base_name = os.path.basename(original_filepath)
    dir_name = os.path.dirname(original_filepath)
    output_filename = os.path.join(dir_name, base_name.replace('.csv', '_with_labels.csv'))
    print(f"合并完成。正在保存到: {output_filename}")
    merged_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

    marked_rows = merged_df['Status_Code'].notna().sum()
    print(f"\n--- 成功 ---")
    print(f"已将 {marked_rows} 个工况标签成功标记到原始数据中。")
    print(f"(这 {marked_rows} 行, 对应 {len(lookup_df)} 个有效窗口的起始点)")
    print(f"文件已保存为: {output_filename}")

if __name__ == '__main__':
    filepath = 'data/cleaned_VHR_data4.0.csv'

    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 200)

    WINDOW_SECONDS = 60
    STEP_SECONDS = 10

    if not os.path.exists(filepath):
        print(f"错误：找不到数据文件: {filepath}")
        print("请将数据文件放在正确的位置，或修改 'filepath' 变量。")
    else:
        features_df = feature_engineering_sliding_window(
            filepath,
            window_seconds=WINDOW_SECONDS,
            step_seconds=STEP_SECONDS
        )

        if features_df is not None and not features_df.empty:
            result_df, feature_cols, scaler ,kmeans = train_clustering_model(features_df)

            if result_df is not None:
                labeled_result_df = analyze_and_visualize(result_df, feature_cols, scaler,kmeans)

                if labeled_result_df is not None:
                    map_labels_to_original_file(filepath, labeled_result_df)
                else:
                    print("未完成状态标注，已跳过合并到原始文件。")

        else:

            print("没有有效的驾驶数据窗口可供分析。")
