## 1.data_processing.py
此文件用于原始VHR数据的预处理，包括以下流程：
读入数据，以collectTime为基准转换时间格式并排序
异常数据处理（尝试了Z-score、IQR；分别删除10436行数据及14209行“异常数据”明显不正确）
缺失数据填充
最后采用Q10-Q90区间，减少了数据删除量，一共删除2506行
## 2.feature_calculate.py
特征工程，在滑动窗口内计算所有特征的均值，标准差，最大最小值并保存为新的文件
## 3.run_cluster.py
对计算的特征值进行PCA降维
分别计算降维维度N，聚类算法（k-means，GMM），以及聚类个数K在什么时候出现最好的聚类效果，选用相应的选项
为经过特征工程计算的数据打上聚类标签
## 4.llmbased_labled.py
计算所有类别的平均特征，以便LLM分析
##5.cluster_to_json.py
将上一步中得到的csv文件转化为每一个类别以及特征数据的json文件，便于写入prompt进行LLM标注
##6.使用设计好的prompt，输入LLM对数据进行标注
##7.merge_situation_labels.py
将LLM标注的标签标回源数据
##8.im_nan.py、sa_fea.py
进行特征重要性分析以及态势与特征的交互分析
##9.ploty_html.py
绘制态势及特征曲线图




