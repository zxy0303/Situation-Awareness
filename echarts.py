import pandas as pd
import json
from pyecharts import options as opts
from pyecharts.charts import Line
from pyecharts.globals import CurrentConfig

file_path = 'all_samples_with_status_labels_k7.csv'  # 数据文件路径
new_data = pd.read_csv(file_path)
new_data['start_time'] = pd.to_datetime(new_data['start_time_int'], unit='ms')
time_series_data_new = new_data[['start_time', 'Status_Code']].dropna()

data = [
    [row['start_time'].strftime('%Y-%m-%d %H:%M:%S'), row['Status_Code']]
    for index, row in time_series_data_new.iterrows()
]

line_chart = (
    Line()
    .add_xaxis([item[0] for item in data])  # 时间轴
    .add_yaxis("状态码", [item[1] for item in data], is_smooth=True)  # 状态码
    .set_global_opts(
        title_opts=opts.TitleOpts(title="车辆状态码随时间变化"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        xaxis_opts=opts.AxisOpts(type_="time", name="时间", axislabel_opts=opts.LabelOpts(formatter="{yyyy}-{MM}-{dd} {HH}:{mm}:{ss}")),
        yaxis_opts=opts.AxisOpts(type_="value", name="状态码"),
        datazoom_opts=opts.DataZoomOpts(type_="slider", range_start=0, range_end=100)  # 拉伸功能
    )
)

line_chart.render("status_code_time_series.html")

