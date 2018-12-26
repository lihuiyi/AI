# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from dnn_estimator.csv_preprocessing import preprocessing_csv



columns_ = "columns"
record_defaults_ = "record_defaults"
label_name_ = "label_name"



def decode_csv(line , pkl_obj):
    columns = pkl_obj[columns_]
    label_name = pkl_obj[label_name_]
    record_defaults = pkl_obj[record_defaults_]
    parsed_line = tf.decode_csv(line , record_defaults)
    features = dict(zip(columns , parsed_line))
    label = features.pop(label_name)
    return features , label



def map_fn(line , pkl_obj):
    feature , label = decode_csv(line, pkl_obj)
    feature, label = preprocessing_csv(feature, label, pkl_obj)
    return feature , label



def input_fn(is_training , csv_path , pkl_obj, shuffle_buffer , batch_size , num_epochs=1):
    # 获取文件名列表
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    # 解码csv，然后数据预处理
    dataset = dataset.map(lambda line: map_fn(line, pkl_obj))
    # 如果是训练模式，就 shuffle 数据，随机种子为 1
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=tf.set_random_seed(1))
        # if use_seed:
        #     dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=tf.set_random_seed(1))
        # else:
        #     dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    # 重复多少个 epochs，然后组成 batch 数据
    dataset = dataset.repeat(num_epochs).batch(batch_size)
    return dataset



def train_input_fn(csv_path, pkl_obj, batch_size, num_epochs):
    return input_fn(
        is_training=True,
        csv_path=csv_path,
        pkl_obj=pkl_obj ,
        shuffle_buffer=1000 ,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )



def eval_input_fn(csv_path, pkl_obj, batch_size, num_epochs=1):
    return input_fn(
        is_training=False,
        csv_path=csv_path,
        pkl_obj=pkl_obj,
        shuffle_buffer=1000 ,
        batch_size=batch_size,
        num_epochs=num_epochs,
    )



def predict_input_fn(features, labels, pkl_obj, batch_size):
    # 把 list 格式的 features 转换矩阵格式
    features = np.array(features)
    if len(features.shape) == 1:
        features = features.reshape(features.shape[0], -1)
    else:
        features = features.T
    # 把矩阵格式的 features 转换为字典格式
    columns = pkl_obj[columns_][0:-1]
    features = dict(zip(columns, features))
    # 数据预处理，features 必须是字典格式
    features, labels = preprocessing_csv(features, labels, pkl_obj)
    # dataset 数据集
    dataset = tf.data.Dataset.from_tensor_slices(features)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    return dataset

