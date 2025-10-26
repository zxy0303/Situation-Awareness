import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

## 此文件仅用于聚类，帮助找到最佳聚类个数，并得到聚类特征
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def feature_engineering_sliding_window(filepath, trip_split_minutes=10, window_seconds=60, step_seconds=10):
    print("开始加载数据...")
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, low_memory=False, encoding='gbk')

    print("数据加载完毕，开始进行特征工程...")

    print("正在转换时间戳格式...")
    df['receiveTime'] = pd.to_datetime(df['receiveTime'], unit='ms')
    #df['receiveTime'] = pd.to_datetime(df['receiveTime'], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai')
    df = df.sort_values('receiveTime').reset_index(drop=True)
    time_diff = df['receiveTime'].diff().dt.total_seconds() / 60
    df['trip_id'] = (time_diff > trip_split_minutes).cumsum()


    window_features = []

    for trip_id, trip_data in df.groupby('trip_id'):

        if trip_data['vehicleSpeed'].max() == 0:
            continue

        trip_data = trip_data.reset_index(drop=True)
        start_time = trip_data['receiveTime'].min()
        end_time = trip_data['receiveTime'].max()

        print(f"\n正在处理行程 {trip_id} (时长: {end_time - start_time})...")
        current_time = start_time
        pbar = tqdm(total=(end_time - start_time).total_seconds())
        while current_time + pd.Timedelta(seconds=window_seconds) <= end_time:
            window_start = current_time
            window_end = current_time + pd.Timedelta(seconds=window_seconds)

            window_df = trip_data[(trip_data['receiveTime'] >= window_start) & (trip_data['receiveTime'] < window_end)]
            if len(window_df) > 5:
                mileage_covered = window_df['totalMileage'].iloc[-1] - window_df['totalMileage'].iloc[0]
                if mileage_covered < 0: mileage_covered = 0 

                window_features.append({
                    'avg_speed': window_df['vehicleSpeed'].mean(),#平均速度
                    'max_speed': window_df['vehicleSpeed'].max(),#最大速度
                    'std_speed': window_df['vehicleSpeed'].std(),#速度标准差
                    'mileage_covered': mileage_covered,#行驶距离
                    'avg_actGearSt': window_df['actGearSt'].mean(),#平均挡位
                    'max_actGearSt': window_df['actGearSt'].max(),#最大挡位
                    'std_actGearSt': window_df['actGearSt'].std(),#挡位标准差
                    'avg_SOC': window_df['battSocAct'].mean(),#平均电量
                    'avg_outsidetem': window_df['outsideAmbTemp'].mean(),#平均室外温度
                    'avg_accmode': window_df['accMode'].mean(),#车辆加速模式
                    'avg_driveMode': window_df['driveMode'].mean(),#平均车辆模式
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
                    'avg_accpedalAct':window_df['accPedalActualPosition'].mean(),#平均踏板位置
                    'max_accpedalAct': window_df['accPedalActualPosition'].max(),
                    'std_accpedalAct': window_df['accPedalActualPosition'].std(),
                    'avg_longAcceleration': window_df['longAcceleration'].mean(),#纵向加速度
                    'std_longAcceleration': window_df['longAcceleration'].std(),
                    'avg_lateralAcceleration': window_df['lateralAcceleration'].mean(),#横向加速度
                    'std_lateralAcceleration': window_df['lateralAcceleration'].std(),
                    'avg_actHSRelHum': window_df['actHSRelHum'].mean(),#车内湿度
                })

            current_time += pd.Timedelta(seconds=step_seconds)
            pbar.update(step_seconds)
        pbar.close()


    features_df = pd.DataFrame(window_features).dropna()
    print(f"\n特征工程完成，共从所有行程中提取出 {len(features_df)} 个有效的时间窗口。")
    try:
        print(f"正在将特征文件保存到:")
        features_df.to_csv('./result.csv', index=False, encoding='utf-8-sig')
        print("文件保存成功！")
    except Exception as e:
        print(f"文件保存失败: {e}")
    return features_df

def train_clustering_model(features_df):
    if features_df is None or features_df.empty:
        print("特征数据为空，无法进行聚类。")
        return None, None, None
    print("\n开始训练聚类模型...")
    feature_cols = [col for col in features_df.columns if col not in ['start_time']]
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
    plt.show()

    optimal_k = int(input("根据肘部法则图，请输入您认为的最佳K值: "))

    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    features_df['cluster'] = kmeans.fit_predict(X_scaled)
    print(f"模型训练完成，已将数据聚为 {optimal_k} 类。")
    return features_df, feature_cols, scaler,kmeans


def analyze_and_visualize(result_df, feature_cols, scaler,kmeans):
    if result_df is None:
        return
    print("\n--- 聚类结果分析 ---")

    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers_real = pd.DataFrame(scaler.inverse_transform(cluster_centers_scaled), columns=feature_cols,
                                        index=range(kmeans.n_clusters))

    print("每个类别的窗口数量:")
    print(result_df['cluster'].value_counts())
    print("\n每个类别的特征均值 (真实类别数值):")
    print(cluster_centers_real.round(2))
    k=result_df['cluster'].nunique()
    cluster_centers_real.round(2).to_csv(f'cluster_centers_analysis_{k}.csv', encoding='utf-8-sig')
    print(f"\n聚类中心特征均值已保存为 cluster_centers_analysis_{k}.csv")

    labels = np.array(feature_cols)
    n_labels = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_labels, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, polar=True)

    for i, stats in enumerate(cluster_centers_scaled):
        count = result_df['cluster'].value_counts().get(i, 0)
        stats = np.concatenate((stats, [stats[0]]))
        ax.plot(angles, stats, label=f"类别 {i} (数量: {count})")
        ax.fill(angles, stats, alpha=0.1)
    # for i, row in cluster_centers_scaled.iterrows():
    #     stats = row.values
    #     stats = np.concatenate((stats, [stats[0]]))
    #     ax.plot(angles, stats, label=f"类别 {i} (数量: {result_df['cluster'].value_counts()[i]})")
    #     ax.fill(angles, stats, alpha=0.1)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title("不同驾驶态势聚类结果雷达图 (标准化后)")
    ax.legend()
    k=result_df['cluster'].nunique()
    plt.savefig(f'radar_chart_{k}.png', dpi=300)
    plt.show()
    print(f"雷达图已保存为 radar_chart_{k}.png")


if __name__ == '__main__':
    filepath = 'data/cleaned_VHR_data3.0.csv'
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 200)
    WINDOW_SECONDS = 60
    STEP_SECONDS = 10

    features_df = feature_engineering_sliding_window(
        filepath,
        window_seconds=WINDOW_SECONDS,
        step_seconds=STEP_SECONDS
    )


    if features_df is not None and not features_df.empty:
        result_df, feature_cols, scaler ,kmeans= train_clustering_model(features_df)


        if result_df is not None:
            analyze_and_visualize(result_df, feature_cols, scaler,kmeans)
    else:

        print("没有有效的驾驶数据窗口可供分析。")

