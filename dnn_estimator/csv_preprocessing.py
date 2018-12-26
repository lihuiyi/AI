# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import tensorflow as tf



# 分类还是回归
classifier_or_regressor_ = "classifier_or_regressor"
# 列
columns_ = "columns"
record_defaults_ = "record_defaults"
# 标签预处理
label_preprocessing_method = "label_preprocessing_method"
label_standardScaler_obj = "label_standardScaler_obj"
label_unique_value_list_ = "label_unique_value_list"
# 特征离散化
features_discretization_columns_ = "features_discretization_columns"
discretization_bins = "discretization_bins"
# 特征哑编码
features_onehot_columns_ = "features_onehot_columns"
features_unique_value_list_ = "features_unique_value_list"
# 特征标准化
features_standardScaler_columns = "features_standardScaler_columns"
features_standardScaler_obj = "features_standardScaler_obj"



def load_pkl(pkl_Path):
    with open(pkl_Path , 'rb') as f:
        pkl_obj = pickle.load(f)
    return pkl_obj



def feature_standardScaler(feature, pkl_obj):
    columns = pkl_obj[features_standardScaler_columns]
    mean = pkl_obj[features_standardScaler_obj].mean_
    std = pkl_obj[features_standardScaler_obj].scale_
    if type(columns) is not list:
        columns = [columns]
    # 对每一列分别进行标准化
    for i in range(len(columns)):
        feature_name = columns[i]
        feature[feature_name] = tf.cast(feature[feature_name], tf.float64, name="float32_to_float64_" + str(i))
        feature[feature_name] = (feature[feature_name] - mean[i]) / std[i]  # 减均值，除标准差
        feature[feature_name] = tf.cast(feature[feature_name], tf.float32)
    return feature



def label_standardScaler(label, pkl_obj):
    mean = pkl_obj[label_standardScaler_obj].mean_
    std = pkl_obj[label_standardScaler_obj].scale_
    label = tf.cast(label, tf.float64, name="float32_to_float64")
    label = (label - mean) / std  # 减均值，除标准差
    label = tf.cast(label, tf.float32)
    return label



def preprocessing_csv(feature, label, pkl_obj):
    if pkl_obj[features_standardScaler_columns] is not None:
        feature = feature_standardScaler(feature, pkl_obj)
    if (label is not None) and (pkl_obj[label_preprocessing_method] == "标准化"):
        label = label_standardScaler(label, pkl_obj)
    return feature, label



def bucketized_column(pkl_obj , use_embedding_column=False):
    """
    功能：分桶列。特征离散化，然后哑编码
    :param use_embedding_column:
    :return:
    """
    columns = pkl_obj[features_discretization_columns_]
    bins = pkl_obj[discretization_bins]
    if type(columns) is not list:
        columns = [columns]
    if type(bins[0]) is not list:
        bins = [bins]
    # 开始分桶
    bucketized_column_list = []
    for i in range(len(columns)):
        feature_name = columns[i]
        numeric_column = tf.feature_column.numeric_column(feature_name)
        bucketized_column = tf.feature_column.bucketized_column(source_column=numeric_column, boundaries=bins[i])
        if use_embedding_column:
            # 嵌入列的维度
            number_of_categories = len(bins[i]) + 1
            if (number_of_categories**0.25) < 3:
                embedding_dimension = 3
            else:
                embedding_dimension = round((number_of_categories**0.25))
            # 嵌入列
            dense_column = tf.feature_column.embedding_column(
                categorical_column=bucketized_column, dimension=embedding_dimension
            )
        else:
            # 指标列
            dense_column = tf.feature_column.indicator_column(bucketized_column)
        bucketized_column_list.append(dense_column)
    return bucketized_column_list



def categorical_column(pkl_obj, dimension_threshold=10):
    """
    功能：分类列。特征哑编码。
    分类哈希列只能对字符串进行哑编码，其他分类列可以对字符串和数字进行哑编码
    """
    columns = pkl_obj[columns_]
    record_defaults = pkl_obj[record_defaults_]
    onehot_columns = pkl_obj[features_onehot_columns_]
    features_unique_value_list = pkl_obj[features_unique_value_list_]
    if type(onehot_columns) is not list:
        onehot_columns = [onehot_columns]
    # 构建哑编码列对应的 record_defaults
    record_defaults_dict = dict(zip(columns, record_defaults))
    # 开始哑编码
    categorical_column_list = []
    for i in range(len(onehot_columns)):
        feature_name = onehot_columns[i]
        data_type = type(record_defaults_dict[feature_name][0])
        features_unique_value = features_unique_value_list[i]
        # 如果当前这一列的数据类型是 str，那么调用 single_categorical_str_column() 函数。否则调用 single_categorical_int_column()
        if data_type is str:
            dense_column = single_categorical_str_column(feature_name, features_unique_value, dimension_threshold)
        else:
            dense_column = single_categorical_int_column(feature_name, features_unique_value, dimension_threshold)
        categorical_column_list.append(dense_column)
    return categorical_column_list



def single_categorical_str_column(feature_name, features_unique_value, dimension_threshold):
    onehot_dimension = len(features_unique_value)
    hash_bucket_size = round(onehot_dimension * (2 / 3))
    # 如果哑编码小于等于10列，就用 分类词汇列 + 指标列
    if onehot_dimension <= dimension_threshold:
        # 分类词汇列
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key=feature_name, vocabulary_list=features_unique_value
        )
        # 指标列
        dense_column = tf.feature_column.indicator_column(categorical_column)
    # 如果哑编码大于10列，并且分类哈希列的size小于等于10列，就用 分类哈希列 + 指标列，分类哈希列的size = 哑编码列数 * (2/3)
    # 如果哑编码大于10列，并且分类哈希列的size大于10列，就用 分类词汇列 + 嵌入列，嵌入列的size从 3 起步
    else:
        # 分类哈希列的size小于等于10列
        if hash_bucket_size <= dimension_threshold:
            # 分类哈希列
            categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
                key=feature_name, hash_bucket_size=hash_bucket_size
            )
            # 指标列
            dense_column = tf.feature_column.indicator_column(categorical_column)
        # 分类哈希列的size大于8列
        else:
            # 分类词汇列
            categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key=feature_name, vocabulary_list=features_unique_value
            )
            # 嵌入列
            embedding_dimension = embedding_column_dimension(onehot_dimension)  # 嵌入列的维度
            dense_column = tf.feature_column.embedding_column(
                categorical_column=categorical_column, dimension=embedding_dimension
            )
    return dense_column



def single_categorical_int_column(feature_name, features_unique_value, dimension_threshold):
    onehot_dimension = len(features_unique_value)
    # 分类词汇列
    categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature_name, vocabulary_list=features_unique_value
    )
    # 如果哑编码小于等于10列，就用指标列
    # 如果哑编码大于10列，就用嵌入列
    if onehot_dimension <= dimension_threshold:
        # 指标列
        dense_column = tf.feature_column.indicator_column(categorical_column)
    else:
        # 嵌入列
        embedding_dimension = embedding_column_dimension(onehot_dimension)  # 嵌入列的维度
        dense_column = tf.feature_column.embedding_column(
            categorical_column=categorical_column, dimension=embedding_dimension
        )
    return dense_column



def embedding_column_dimension(onehot_dimension):
    if (onehot_dimension ** 0.25) < 3:
        embedding_dimension = 3
    else:
        embedding_dimension = round((onehot_dimension ** 0.25))
    return embedding_dimension



def numeric_column(pkl_obj):
    columns = pkl_obj[columns_][0:-1]  # 训练集中全部特征的列名
    record_defaults = pkl_obj[record_defaults_][0:-1]  # 训练集中每一列对应的数据类型、默认值。例如：[[0.0], [0.0], [""]]
    features_discretization_columns = pkl_obj[features_discretization_columns_]  # 特征离散化的列名
    if type(columns) is not list:
        columns = [columns]
    if type(features_discretization_columns) is not list:
        features_discretization_columns = [features_discretization_columns]
    # 把 record_defaults 转换为 DataFrame。假如转换以前 shape=(60,1)，转换以后 shape=(1,60)
    record_defaults = np.array(record_defaults).T
    record_defaults = pd.DataFrame(record_defaults, columns=columns)
    # 根据 record_defaults 中的数据类型，得到 tensorflow 全部数值列的列名
    all_numeric_column_name = []
    for column in columns:
        f64 = record_defaults[column].dtypes == "float64"
        f32 = record_defaults[column].dtypes == "float32"
        i64 = record_defaults[column].dtypes == "int64"
        i32 = record_defaults[column].dtypes == "int32"
        if f64 or f32 or i64 or i32:  # 只要满足其中一个条件，就是数值列
            if column not in features_discretization_columns:  # 排除特征离散化的列，因为离散化是分桶列，不是数值列
                all_numeric_column_name.append(column)
    # 创建 tensorflow 数值列
    numeric_column_list = []
    for feature_name in all_numeric_column_name:
        numeric_column_list.append(tf.feature_column.numeric_column(key=feature_name))
    return numeric_column_list



def get_feature_column(pkl_obj):
    my_feature_column = []
    # 分桶列
    if pkl_obj[features_discretization_columns_] is not None:
        bucketized_column_list = bucketized_column(pkl_obj , use_embedding_column=False)
        my_feature_column.extend(bucketized_column_list)
    # 分类列
    if pkl_obj[features_onehot_columns_] is not None:
        categorical_column_list = categorical_column(pkl_obj)
        my_feature_column.extend(categorical_column_list)
    # 数值列
    numeric_column_list = numeric_column(pkl_obj)
    my_feature_column.extend(numeric_column_list)
    return my_feature_column



def get_label_vocabulary(pkl_obj):
    classifier_or_regressor = pkl_obj[classifier_or_regressor_]
    # 读取 label 不重复值的集合
    label_unique_value_list = pkl_obj[label_unique_value_list_]
    # 确定 n_classes
    n_classes = len(label_unique_value_list)
    # 根据不同的情况确定 label_vocabulary
    if (classifier_or_regressor == "分类") or (classifier_or_regressor == "classifier"):
        if type(label_unique_value_list[0]) is str:
            label_vocabulary = label_unique_value_list
        else:
            label_vocabulary = None
    else:
        label_vocabulary = None
    return label_vocabulary, n_classes

