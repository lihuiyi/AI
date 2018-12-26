# -*- coding: utf-8 -*-

from dnn_estimator.csv_split import Data_Split



# csv_path = r"F:\data\岩石分类\岩石分类.csv"
# chunksize = 200000
# test_size = 0.2
# validation_size = 0.2
# # 分类还是回归
# classifier_or_regressor = "分类"  # "分类"、"回归"
# # 标签预处理
# label_name = "label"
# label_preprocessing_method = None  # "标准化" 或者 None
# # 特征离散化
# features_discretization_columns = None
# discretization_bins = None
# # 特征哑编码
# features_onehot_columns = None
# # 特征标准化
# a = []
# for i in range(60):
#     col = "feature" + str(i)
#     a.append(col)
# features_standardScaler_columns = a






csv_path = r"C:\Users\lenovo\Desktop\信用卡欺诈检测\信用卡欺诈检测.csv"
chunksize = 200000
test_size = 0.2
validation_size = 0.2
# 分类还是回归
classifier_or_regressor = "分类"  # "分类"、"回归"
# 标签预处理
label_name = "Class"
label_preprocessing_method = None  # "标准化" 或者 None
# 特征离散化
features_discretization_columns = None
discretization_bins = None
# 特征哑编码
features_onehot_columns = None
# 特征标准化
features_standardScaler_columns = ["Amount"]




data_split = Data_Split(
    classifier_or_regressor, label_name, label_preprocessing_method, features_discretization_columns, discretization_bins,
    features_onehot_columns, features_standardScaler_columns
)
data_split.fit(csv_path , chunksize , test_size , validation_size)

