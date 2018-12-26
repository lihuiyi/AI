# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn.model_selection import train_test_split  # 分割训练集与测试集
from utils.preprocessing import Data_preprocessing
import warnings
warnings.filterwarnings("ignore")



# # 航班_10期
# # 文件路径
# data_dir = "F:\data\航班\航班_10期"
# file_path = os.path.join(data_dir, "航班_10期.csv")
# # 分割训练集、验证集、测试集
# test_size = 0.2
# eval_size = 0.2
# # 数据预处理参数
# label_name = "Y"
# label_preprocessing_method = "StandardScaler"  # 标签预处理方式，"ToNum"、"Onehot"、"StandardScaler"
# features_discretization_columns = None  # 特征离散化列
# discretization_bins = None  # 特征离散化 bins ,是 list of list
# features_onehot_columns = None  # 特征哑编码列
# features_standardScaler_columns = []  # 特征标准化列
# for i in range(10):
#     features_standardScaler_columns.append(str(i+1) + "期_X")
# features_minMaxScaler_columns = None  # 特征区间缩放列
# features_normalizer_columns = None  # 特征归一化列




# mnist
# 文件路径
data_dir = r"C:\Users\lenovo\Desktop\新建文件夹"
file_path = os.path.join(data_dir, "mnist.csv")
# 分割训练集、验证集、测试集
test_size = 0.2
eval_size = 0.2
# 数据预处理参数
label_name = "label"
label_preprocessing_method = "Onehot"  # 标签预处理方式，"ToNum"、"Onehot"、"StandardScaler"
features_discretization_columns = None  # 特征离散化列
discretization_bins = None  # 特征离散化 bins ,是 list of list
features_onehot_columns = None  # 特征哑编码列
features_standardScaler_columns = None  # 特征标准化列
features_minMaxScaler_columns = None  # 特征区间缩放列
features_normalizer_columns = None  # 特征归一化列



# 读取数据，代码不用改
with open(file_path) as f:
    dataset = pd.read_csv(f)
# 分割训练集、验证集、测试集
train_data, test_data = train_test_split(dataset, test_size=test_size, shuffle=True, random_state=42)
train_data, eval_data = train_test_split(train_data, test_size=eval_size, shuffle=True, random_state=42)
# 数据预处理
dp = Data_preprocessing()  # 实列化数据预处理对象
train_data = dp.fit_transform(
    data_dir, train_data, label_name, label_preprocessing_method,
    features_discretization_columns, discretization_bins, features_onehot_columns,
    features_standardScaler_columns, features_minMaxScaler_columns, features_normalizer_columns
)
eval_data = dp.transform(eval_data, data_dir)
test_data = dp.transform(test_data, data_dir)
# 把预处理后的数据保存到 csv 文件
train_dir = os.path.join(data_dir, "train_data")
is_exists = os.path.exists(train_dir)  # 判断一个目录是否存在
if is_exists is False:
    os.makedirs(train_dir)  # 创建目录
train_data.to_csv(os.path.join(train_dir, "train.csv"), index=False, encoding="gbk")
eval_data.to_csv(os.path.join(train_dir, "eval.csv"), index=False, encoding="gbk")
test_data.to_csv(os.path.join(train_dir, "test.csv"), index=False, encoding="gbk")
