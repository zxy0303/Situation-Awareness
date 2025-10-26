import pandas as pd

file_path = 'data/2025-08-23.csv'
try:
    df = pd.read_csv(file_path, low_memory=False)
except UnicodeDecodeError:
    df = pd.read_csv(file_path, low_memory=False, encoding='gbk')
#df['receiveTime'] = pd.to_datetime(df['receiveTime'], unit='ms')
df = df.sort_values('receiveTime').reset_index(drop=True)

columns_to_drop = [
    'vin', 'vehicleSeries', 'vehicleModel', 'receiveTime', 'isAlarm', 'isIsolate',
    'resendFlag', 'reportTime', 'partition_key',
    'mSleConnectStChann1', 'mSleConnectTyChann1', 'mSleRunningStChann1', 'mSlePacketLossRatChann1',
    'mSleConnectStChann2', 'mSleConnectTyChann2', 'mSleRunningStChann2', 'mSlePacketLossRatChann2',
    'mSleConnectStChann3', 'mSleConnectTyChann3', 'mSleRunningStChann3', 'mSlePacketLossRatChann3',
    'mSleConnectStChann4', 'mSleConnectTyChann4', 'mSleRunningStChann4', 'mSlePacketLossRatChann4'
]
data_cleaned = df.drop(columns=columns_to_drop)
data_cleaned = data_cleaned.dropna(axis=1, how='all')

data_cleaned = data_cleaned.fillna(method='bfill')
data_cleaned = data_cleaned.fillna(method='ffill')
print(data_cleaned.columns)

new_file_path = 'data/cleaned_VHR_data4.0.csv'
data_cleaned.to_csv(new_file_path, index=False)

print(f"清洗后的数据已保存为：{new_file_path}")

