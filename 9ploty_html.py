import pandas as pd
import json
import plotly.graph_objects as go

# 1. 读取文件
try:
    with open('all_features_combined.json', 'r', encoding='utf-8') as f:
        all_features = json.load(f)
except FileNotFoundError:
    # 备用：如果没有json文件，尝试从df列名获取
    all_features = []

try:
    df = pd.read_excel('labeled_source_data_with_situations.xlsx')
except FileNotFoundError:
    print("错误: 找不到文件")
    df = pd.DataFrame()

if not df.empty:
    # 数据转换
    df['timestamp'] = pd.to_datetime(df['collectTime_dt'])

    # -------------------------------------------------------------
    # 【新增修改】: 增加 8 小时时差调整为北京时间
    # -------------------------------------------------------------
    df['timestamp'] = df['timestamp'] + pd.Timedelta(hours=8)

    # 自动识别标签列（假设最后8列）
    labels_columns = df.columns[-9:]

    # 如果没有读取到 all_features，则排除掉非特征列
    if not all_features:
        exclude_cols = set(list(labels_columns) + ['collectTime_dt', 'timestamp', 'Unnamed: 0'])
        all_features = [c for c in df.columns if c not in exclude_cols]


    # --- 构建标签文本 ---
    def create_hover_text(row):
        lines = []
        # 标题也可以改一下，提示这是北京时间
        lines.append("<b>▼ 当前时刻态势 (北京时间) ▼</b>")
        for col in labels_columns:
            lines.append(f"{col}: {row[col]}")
        return "<br>".join(lines)


    df['label_text'] = df.apply(create_hover_text, axis=1)

    # 创建画布
    fig = go.Figure()

    # --- 添加特征曲线 ---
    for i, feature in enumerate(all_features):
        if feature in df.columns:
            visible_status = True if i < 5 else 'legendonly'

            fig.add_trace(go.Scatter(
                x=df['timestamp'],  # 此时已经是北京时间
                y=df[feature],
                name=feature,
                mode='lines',
                visible=visible_status,
                hovertemplate="%{y}"
            ))

    # --- 添加“态势信息层” ---
    reference_y = df[all_features[0]] if all_features else df.iloc[:, 0]

    fig.add_trace(go.Scatter(
        x=df['timestamp'],  # 此时已经是北京时间
        y=reference_y,
        name="态势详情",
        mode='lines',
        line=dict(width=0),
        opacity=0,
        showlegend=True,
        customdata=df['label_text'],
        hovertemplate="%{customdata}<extra></extra>"
    ))

    # --- 布局设置 ---
    fig.update_layout(
        title="实时特征与态势标签分析 (北京时间 UTC+8)",  # 标题更新

        hovermode="x unified",

        xaxis=dict(
            title="时间 (Beijing Time)",  # 坐标轴更新
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(
            title="特征数值",
            fixedrange=False
        ),
        legend=dict(
            title="点击图例显示/隐藏",
            x=1.01, y=1
        ),
        template="plotly_white"
    )

    # 保存
    output_file = "situational_analysis_beijing_time.html"
    fig.write_html(output_file)
    print(f"生成成功！文件已保存为: {output_file}")
    print("时间戳已成功向后推移 8 小时。")

else:
    print("数据为空，请检查文件路径。")