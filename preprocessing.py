# 导入需要的包
import pandas as pd

# 读取数据，指定跳过第一行
file_path = 'data/2025-08-23.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, low_memory=False, encoding='gbk')
#df['receiveTime'] = pd.to_datetime(df['receiveTime'], unit='ms')
df = df.sort_values('receiveTime').reset_index(drop=True)

# 定义需要删除的列
columns_to_drop = [
    'vin', 'vehicleSeries', 'vehicleModel', 'receiveTime', 'isAlarm', 'isIsolate',
    'resendFlag', 'reportTime', 'partition_key',
    'mSleConnectStChann1', 'mSleConnectTyChann1', 'mSleRunningStChann1', 'mSlePacketLossRatChann1',
    'mSleConnectStChann2', 'mSleConnectTyChann2', 'mSleRunningStChann2', 'mSlePacketLossRatChann2',
    'mSleConnectStChann3', 'mSleConnectTyChann3', 'mSleRunningStChann3', 'mSlePacketLossRatChann3',
    'mSleConnectStChann4', 'mSleConnectTyChann4', 'mSleRunningStChann4', 'mSlePacketLossRatChann4'
]

# 删除这些列
data_cleaned = df.drop(columns=columns_to_drop)

# 删除全为空的列
data_cleaned = data_cleaned.dropna(axis=1, how='all')
# 对时间序列数据进行后向填充
data_cleaned = data_cleaned.fillna(method='bfill')
data_cleaned = data_cleaned.fillna(method='ffill')

# 查看清洗后的列名
print(data_cleaned.columns)

# 设置新文件保存路径
new_file_path = 'data/cleaned_VHR_data4.0.csv'

# 保存清洗后的数据到新文件
data_cleaned.to_csv(new_file_path, index=False)

# 输出新文件路径
print(f"清洗后的数据已保存为：{new_file_path}")
