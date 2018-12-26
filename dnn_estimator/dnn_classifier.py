# -*- coding: utf-8 -*-

import os
import shutil
import pickle
import tensorflow as tf
from dnn_estimator.csv_preprocessing import load_pkl, get_feature_column, get_label_vocabulary
from dnn_estimator.csv_pipeline import train_input_fn , eval_input_fn , predict_input_fn



processing_pkl_name = "数据预处理.pkl"
hyper_parameter_pkl_name = "神经网络超参数.pkl"
hidden_units_ = "hidden_units_"
dropout_rate_ = "dropout_rate_"



def get_model_hyper_parameter(hidden_units, dropout_rate):
    parameter_pkl_obj = dict(
        hidden_units_ = hidden_units,
        dropout_rate_ = dropout_rate
    )
    return parameter_pkl_obj



def save_pkl(model_dir, processing_pkl_Path, hidden_units, dropout_rate):
    # 把 数据预处理.pkl 文件复制到 model_dir 目录下
    target_path = os.path.join(model_dir, processing_pkl_name)
    shutil.copyfile(processing_pkl_Path, target_path)  # 复制文件操作
    # 把模型的超参数保存到 pkl 文件中，保存目录是 model_dir。在调用 predict() 函数时，直接加载 pkl文件
    parameter_pkl_obj = get_model_hyper_parameter(hidden_units, dropout_rate)
    pkl_path = os.path.join(model_dir, hyper_parameter_pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(parameter_pkl_obj, f)



def model(hidden_units, model_dir, dropout_rate, pkl_obj):
    tf.logging.set_verbosity(tf.logging.ERROR)  # 设置日志
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # 使用 Winograd 非融合算法可以提供小的性能提升
    # config 配置。使用allow_soft_placement = True，这是多GPU所必需
    session_config = tf.ConfigProto(inter_op_parallelism_threads=0, intra_op_parallelism_threads=0, allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(session_config=session_config)
    # 获取特征列
    my_feature_column = get_feature_column(pkl_obj)
    # 获取 label_vocabulary 和 n_classes
    label_vocabulary, n_classes = get_label_vocabulary(pkl_obj)
    # 实例化 DNNClassifier
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_column,
        hidden_units=hidden_units,
        n_classes=n_classes,
        label_vocabulary=label_vocabulary,
        model_dir=model_dir,
        activation_fn=tf.nn.relu,
        dropout=dropout_rate,
        config=run_config
    )
    return classifier



def fit(train_path, eval_path, test_path, processing_pkl_Path, model_dir, hidden_units, dropout_rate,
        batch_size, train_epochs, epochs_per_eval):
    # 加载预数据处理对象
    processing_pkl_obj = load_pkl(processing_pkl_Path)
    # 实例化模型
    classifier = model(hidden_units, model_dir, dropout_rate, processing_pkl_obj)
    # 训练模型、评估模型
    for i in range(train_epochs // epochs_per_eval):
        # 训练模型
        classifier.train(
            input_fn=lambda: train_input_fn(
                csv_path=train_path, pkl_obj=processing_pkl_obj, batch_size=batch_size, num_epochs=epochs_per_eval
            )
        )
        # 数据处理.pkl 、神经网络超参数.pkl 文件保存到 model_dir 目录下
        if i == 0:
            save_pkl(model_dir, processing_pkl_Path, hidden_units, dropout_rate)
        # 使用训练集评估模型
        train_result = classifier.evaluate(
            input_fn=lambda: train_input_fn(
                csv_path=train_path, pkl_obj=processing_pkl_obj, batch_size=batch_size, num_epochs=1
            ),
            steps=100
        )
        print(train_result["global_step"])
        print("   训练集:", train_result)
        # 使用验证集评估模型
        eval_result = classifier.evaluate(
            input_fn=lambda: eval_input_fn(
                csv_path=eval_path, pkl_obj=processing_pkl_obj, batch_size=batch_size, num_epochs=1
            )
        )
        print("   验证集:" , eval_result, "\n")
    # 使用测试集评估模型
    test_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(
            csv_path=test_path, pkl_obj=processing_pkl_obj, batch_size=batch_size, num_epochs=1
        )
    )
    print("测试集:", test_result, "\n")



def predict(features, model_dir):
    # 数据处理.pkl 文件路径、神经网络超参数.pkl 文件的路径
    processing_pkl_Path = os.path.join(model_dir, processing_pkl_name)
    hyper_parameter_path = os.path.join(model_dir, hyper_parameter_pkl_name)
    # 加载预数据处理对象
    processing_pkl_obj = load_pkl(processing_pkl_Path)
    # 加载模型超参数对象，然后获取模型的超参数
    hyper_parameter_pkl_obj = load_pkl(hyper_parameter_path)
    hidden_units = hyper_parameter_pkl_obj[hidden_units_]
    dropout_rate =hyper_parameter_pkl_obj[dropout_rate_]
    # 得到 label_vocabulary
    label_vocabulary = get_label_vocabulary(processing_pkl_obj)
    # 实例化模型
    classifier = model(hidden_units, model_dir, dropout_rate, processing_pkl_obj)
    # 预测
    predict_result = classifier.predict(
        input_fn=lambda: predict_input_fn(features, labels=None, pkl_obj=processing_pkl_obj, batch_size=128)
    )
    # 解析预测结果
    y_pred_list = []
    for pred_dict in zip(predict_result):
        classes = pred_dict[0]["classes"][0]
        classes = bytes.decode(classes)
        if label_vocabulary is None:
            classes = int(classes)
        y_pred_list.append(classes)
    if len(y_pred_list) == 1:
        y_pred = y_pred_list[0]
    else:
        y_pred = y_pred_list
    return y_pred

