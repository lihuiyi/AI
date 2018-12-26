# -*- coding: utf-8 -*-

import os
from dnn_estimator.csv_preprocessing import load_pkl



# 分类还是回归
classifier_or_regressor_ = "classifier_or_regressor"




# 文件路径
# data_dir = r"F:\data\鸢尾花\train_data"
# train_path = os.path.join(data_dir , "train.csv")
# eval_path = os.path.join(data_dir , "eval.csv")
# test_path = os.path.join(data_dir , "test.csv")
# processing_pkl_Path = os.path.join(data_dir , "数据预处理.pkl")
# 模型超参数
# model_dir = r"F:\model\a"
# train_epochs = 300
# epochs_per_eval = 100
# use_seed = True
# hidden_units =[20 , 20]
# dropout_rate = 0.5
# batch_size = 100



# 文件路径
# data_dir = r"F:\data\岩石分类\train_data"
# train_path = os.path.join(data_dir , "train.csv")
# eval_path = os.path.join(data_dir , "eval.csv")
# test_path = os.path.join(data_dir , "test.csv")
# processing_pkl_Path = os.path.join(data_dir , "数据预处理.pkl")
# 模型超参数
# model_dir = r"F:\model\a"
# train_epochs = 300
# epochs_per_eval = 100
# hidden_units =[60 , 60]
# dropout_rate = 0.5
# batch_size = 128



# 文件路径
data_dir = r"F:\data\波士顿房价\train_data"
train_path = os.path.join(data_dir , "train.csv")
eval_path = os.path.join(data_dir , "eval.csv")
test_path = os.path.join(data_dir , "test.csv")
processing_pkl_Path = os.path.join(data_dir , "数据预处理.pkl")
# 模型超参数
model_dir = r"F:\model\a"
train_epochs = 300
epochs_per_eval = 100
hidden_units =[80 , 40]
dropout_rate = 0.5
batch_size = 128



# 训练模型，代码不用改
processing_pkl_obj = load_pkl(processing_pkl_Path)
classifier_or_regressor = processing_pkl_obj[classifier_or_regressor_]
if (classifier_or_regressor == "分类") or (classifier_or_regressor == "classifier"):
    from dnn_estimator.dnn_classifier import fit, predict
else:
    from dnn_estimator.dnn_regressor import fit, predict
fit(
    train_path, eval_path, test_path, processing_pkl_Path, model_dir, hidden_units, dropout_rate,
    batch_size, train_epochs, epochs_per_eval
)



# 预测
# features = [[4.9, 3 , 1.4, 0.2], [4.9, 3 , 1.4, 0.2]]
# features = [
# 0.026, 0.0192, 0.0254, 0.0061, 0.0352, 0.0701, 0.1263, 0.108, 0.1523, 0.163, 0.103, 0.2187, 0.1542,
#     0.263, 0.294, 0.2978, 0.0699, 0.1401, 0.299, 0.3915, 0.3598, 0.2403, 0.4208, 0.5675, 0.6094, 0.6323,
#     0.6549, 0.7673, 1, 0.8463, 0.5509, 0.4444, 0.5169, 0.4268, 0.1802, 0.0791, 0.0535, 0.1906, 0.2561,
#     0.2153, 0.2769, 0.2841, 0.1733, 0.0815, 0.0335, 0.0933, 0.1018, 0.0309, 0.0208, 0.0318, 0.0132, 0.0118,
#     0.012, 0.0051, 0.007, 0.0015, 0.0035, 0.0008, 0.0044, 0.0077
# ]
features = [[0.09178, 0, 4.05, 0, 0.51, 6.416, 84.1, 2.6463, 5, 296, 16.6, 395.5, 9.04]]

y_pred = predict(features, model_dir)
print("预测结果:\n", y_pred)

