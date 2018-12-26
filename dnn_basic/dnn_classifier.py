# -*- coding: utf-8 -*-

import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf



hyper_parameter_pkl_name = "神经网络超参数.pkl"
input_size_ = "input_size"
hidden_units_ = "hidden_units"
n_classes_ = "n_classes"
keep_prob_ = "keep_prob"
learning_rate_ = "learning_rate"
weight_decay_ = "weight_decay"
batch_size_ = "batch_size"
train_epochs_ = "train_epochs"
epochs_per_eval_ = "epochs_per_eval"




def forward_propagation(features, hidden_units, n_classes, keep_prob_placeholder):
    # 隐藏层
    net = features
    for units in hidden_units:
        net = tf.layers.dense(
            inputs=net,
            units=units,
            activation=tf.nn.relu,
            kernel_initializer=tf.variance_scaling_initializer()
        )
        net = tf.nn.dropout(net, keep_prob_placeholder)
    # 输出层
    logits = tf.layers.dense(
        inputs=net,
        units=n_classes,
        kernel_initializer=tf.variance_scaling_initializer()
    )
    return logits



def backward_propagation(labels, logits, weight_decay, learning_rate, global_step):
    # 定义损失函数
    cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)  # 交叉熵
    l2_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()]
    )
    loss = cross_entropy + l2_loss
    tf.summary.scalar("loss", loss)
    # 定义优化器，然后最小化损失函数
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op



def accuracy_rate(labels, logits):
    """
    功能：获取正确率
    参数：
        labels：真实值
        logits：输出层的输出
    返回值：正确率
    """
    y_pred = tf.argmax(logits, 1)
    labels = tf.argmax(labels, 1)
    correct_prediction = tf.equal(y_pred, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy



def auc(logits, labels):
    auc_value, auc_op = tf.metrics.auc(labels=tf.argmax(labels, 1), predictions=tf.argmax(logits, 1))
    return [auc_value, auc_op]



def accuracy(logits, labels):
    accuracy_value, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(labels, 1), predictions=tf.argmax(logits,1))
    return [accuracy_value, accuracy_op]



def next_batch(current_iteration_num, number_samples, features, labels, batch_size):
    """
    功能：获取当前迭代时的 batch 数据
    参数：
        current_iteration_num：当前是第几次迭代
        number_samples：训练样本的数量
        features：特征
        labels：标签
        batch_size：就是 batch_size
    返回值：当前迭代时的 batch 数据
    """
    start_index = (current_iteration_num * batch_size) % number_samples #当前batch的起始index
    end_index = min(start_index + batch_size  , number_samples) #当前batch的结束index
    batch_x = features[start_index: end_index] #当前batch的起始数据
    batch_y = labels[start_index: end_index] #当前batch的结束数据
    return batch_x , batch_y



def train(train_op, accuracy, features, labels, logits, x_train, y_train, x_eval, y_eval, train_epochs, number_samples,
          batch_size, keep_prob_placeholder, keep_prob, epochs_per_eval, model_dir, global_step):
    """
    功能：训练
    参数：
        train_op：就是 train_op
        features：特征，placeholder 类型
        labels：标签，placeholder 类型
        logits：LSTM 的输出值
        x_train：训练集的特征，需要在 sess.run() 时，把特征传进去
        y_train：训练集的标签，需要在 sess.run() 时，把标签传进去
        x_eval：验证集的特征，需要在 sess.run() 时，把特征传进去
        y_eval：验证集的标签，需要在 sess.run() 时，把标签传进去
        train_epochs：训练多少个 epoch
        number_samples：训练集样本数量
        batch_size：就是 batch_size
        keep_prob_placeholder：Dropout 的保留率，placeholder 类型
        keep_prob：Dropout 的保留率，需要在 sess.run() 时，把 keep_prob 传进去
        epochs_per_eval：验证的周期，每迭代多少次使用验证集验证一下，并且保存模型
        model_dir：模型保存的目录
        global_step：就是 global_step
    返回值：没有返回值
    """
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=None)
        # 初始化所有变量
        sess.run(tf.local_variables_initializer())
        init = tf.global_variables_initializer()
        sess.run(init)
        # tensorboard
        writer_train_summary, writer_eval_summary = writer_summary(sess, model_dir)
        tensorboard_summary = tf.summary.merge_all()  # 把代码中全部 tf.summary("某命名空间" , value) 加到 tensorboard_summary 中
        # 迭代次数
        number_iterations = train_epochs * (number_samples / batch_size)
        number_iterations = int(number_iterations)
        # 迭代训练神经网络
        for i in range(number_iterations):
            # 训练
            batch_x, batch_y = next_batch(i, number_samples, x_train, y_train, batch_size)
            sess.run(train_op, feed_dict={features:batch_x , labels:batch_y , keep_prob_placeholder:keep_prob})
            # 验证训练集
            if (i + 1) % epochs_per_eval == 0:
                sess.run(accuracy, feed_dict={features: batch_x, labels: batch_y, keep_prob_placeholder: 1})
                train_accuracy, train_summary = sess.run(
                    [accuracy[0], tensorboard_summary],
                    feed_dict={features: batch_x, labels: batch_y, keep_prob_placeholder: 1}
                )
                writer_train_summary.add_summary(train_summary, i + 1)  # 每 i+1 次 把 summ 加到 tensorboard 中
                # 验证验证集
                sess.run(accuracy, feed_dict={features: x_eval, labels: y_eval, keep_prob_placeholder: 1})
                eval_accuracy, eval_summary = sess.run(
                    [accuracy[0], tensorboard_summary],
                    feed_dict={features: x_eval, labels: y_eval, keep_prob_placeholder: 1}
                )
                writer_eval_summary.add_summary(eval_summary, i + 1)  # 每 i+1 次 把 summ 加到 tensorboard 中
                print(str(i + 1) + "   训练集：" + str(train_accuracy) + "   验证集" + str(eval_accuracy))
                # 保存模型
                save_module(saver, sess, model_dir, global_step)



def save_module(saver , sess , modeldir, global_step, model_name="model.ckpt"):
    """
    功能：保存模型
    参数：
        saver：tf.train.Saver(max_to_keep=None)
        sess：tf.Session() as sess
        modeldir：模型保存的目录
        global_step：就是 global_step
    返回值：没有返回值
    """
    is_exists = os.path.exists(modeldir)  # 判断一个目录是否存在
    if is_exists is False:
        os.makedirs(modeldir)  # 创建目录
    model_path = os.path.join(modeldir, model_name)
    saver.save(sess, model_path , global_step=global_step)



def writer_summary(sess, modeldir):
    """
    功能：定义 tensorboard 中的 writer 对象
    参数：
        sess：tf.Session() as sess
        modeldir：模型保存的目录
    返回值：
        训练集的 writer 对象
        验证集的 writer 对象
    """
    is_exists = os.path.exists(modeldir + r"\eval")  # 判断一个目录是否存在
    if is_exists is False:
        os.makedirs(modeldir + r"\eval")  # 创建目录
    writer_train_summary = tf.summary.FileWriter(modeldir , sess.graph)  # 写入 tensorboard 的核心代码
    writer_eval_summary = tf.summary.FileWriter(modeldir + r"\eval" , sess.graph)  # 写入 tensorboard 的核心代码
    #把 write_data 中的内容写入 .bat 文件
    bat_filePath = modeldir + r"\启动tensorboard.bat"
    write_data = "@echo off\ncd ..\nset pard=%cd%\npopd\ncd %pard%\ntensorboard --logdir=" + modeldir.split("\\")[-1]
    with open(bat_filePath , "w") as file:  # 第一个参数是文件路径。第二个参数是"w"，代表写入。最后赋值给file对象
        file.write(write_data)  # file对象调用write()函数，把变量a中的字符串写入文件
    return writer_train_summary, writer_eval_summary



def save_hyper_parameter(model_dir, input_size, hidden_units, n_classes, keep_prob,
                         learning_rate, weight_decay, batch_size, train_epochs, epochs_per_eval):
    parameter_pkl_obj = {
        input_size_ : input_size,
        hidden_units_ : hidden_units,
        n_classes_ : n_classes,
        keep_prob_ : keep_prob,
        learning_rate_ : learning_rate,
        weight_decay_ : weight_decay,
        batch_size_ : batch_size,
        train_epochs_ : train_epochs,
        epochs_per_eval_ : epochs_per_eval
    }
    pkl_path = os.path.join(model_dir, hyper_parameter_pkl_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(parameter_pkl_obj, f)



def fit(x_train, y_train, x_eval, y_eval, input_size, hidden_units, n_classes, batch_size,
        train_epochs, number_samples, weight_decay, learning_rate, keep_prob, epochs_per_eval, model_dir):
    """
    功能：综合了前面所有函数，只要把参数传进去，就可以开始训练
    参数：
       x_train：训练集的特征，需要在 sess.run() 时，把特征传进去
       y_train：训练集的标签，需要在 sess.run() 时，把标签传进去
       x_eval：验证集的特征，需要在 sess.run() 时，把特征传进去
       y_eval：验证集的标签，需要在 sess.run() 时，把标签传进去
       time_step：时间阶段
       input_size：每一个阶段的输入维度
       n_classes：分类数
       batch_size：就是 batch_size
       lstm_layers_num：隐藏层数
       hidden_layer_units：隐藏层单元数
       train_epochs：训练多少个 epoch
       number_samples：训练集样本数量
       weight_decay：权重衰减系数，也就是 L2 正则化系数
       learning_rate：学习率
       keep_prob：Dropout 的保留率
       eval_cycle：验证的周期，每迭代多少次使用验证集验证一下，并且保存模型
       model_dir：模型保存的目录
    返回值：无
    """
    features = tf.placeholder(tf.float32, [None , input_size])  # [batch_size, 784]
    labels = tf.placeholder(tf.float32, [None , n_classes])  # [batch_size, n_classes]
    keep_prob_placeholder = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
    global_step = tf.train.get_or_create_global_step()
    logits = forward_propagation(
        features, hidden_units, n_classes, keep_prob_placeholder
    )
    # accuracy = accuracy_rate(labels, logits)
    accuracy = auc(logits, labels)

    train_op = backward_propagation(labels, logits, weight_decay, learning_rate, global_step)
    train(
        train_op, accuracy, features, labels, logits, x_train, y_train, x_eval, y_eval, train_epochs, number_samples,
        batch_size, keep_prob_placeholder, keep_prob, epochs_per_eval, model_dir, global_step
    )
    save_hyper_parameter(
        model_dir, input_size, hidden_units, n_classes, keep_prob,
        learning_rate, weight_decay, batch_size, train_epochs, epochs_per_eval
    )







data_dir = r"C:\Users\lenovo\Desktop\信用卡欺诈检测\train_data"
train_path = os.path.join(data_dir, "train.csv")
eval_path = os.path.join(data_dir, "eval.csv")
test_path = os.path.join(data_dir, "test.csv")

with open(train_path) as f:
    train_data = pd.read_csv(f)
with open(eval_path) as f:
    eval_data = pd.read_csv(f)
# with open(test_path) as f:
#     test_data = pd.read_csv(f)



label_name = "Class"
label_preprocessing_method = "Onehot"
features_discretization_columns = None  # 特征离散化列
discretization_bins = None  # 特征离散化 bins ,是 list of list
features_onehot_columns = None  # 特征哑编码列
features_standardScaler_columns = ["Amount"]  # 特征标准化列
features_minMaxScaler_columns = None  # 特征区间缩放列
features_normalizer_columns = None  # 特征归一化列


from utils.preprocessing import Data_preprocessing
# 数据预处理
dp = Data_preprocessing()  # 实列化数据预处理对象
train_data = dp.fit_transform(
    data_dir, train_data, label_name, label_preprocessing_method,
    features_discretization_columns, discretization_bins, features_onehot_columns,
    features_standardScaler_columns, features_minMaxScaler_columns, features_normalizer_columns
)
eval_data = dp.transform(eval_data, data_dir)



x_train = train_data.loc[:, train_data.columns[0:-2]]
y_train = train_data.loc[:, train_data.columns[-2:]]

x_eval = eval_data.loc[:, eval_data.columns[0:-2]]
y_eval = eval_data.loc[:, eval_data.columns[-2:]]


layer_num = 3
input_size = x_train.shape[1]
hidden_units = [128]
n_classes = 2
batch_size = 64
number_samples = 182276
weight_decay = 1e-4
keep_prob = 0.5
learning_rate = 0.001
train_epochs = 10
epochs_per_eval = 1000
model_dir = r"C:\Users\lenovo\Desktop\model"



fit(x_train, y_train, x_eval, y_eval, input_size, hidden_units, n_classes, batch_size,
        train_epochs, number_samples, weight_decay, learning_rate, keep_prob, epochs_per_eval, model_dir)

