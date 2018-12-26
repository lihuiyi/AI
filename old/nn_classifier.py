# -*- coding: utf-8 -*-

import os
import re
import datetime
import tensorflow as tf
from config.nn_config import NN_Config
from config.configparser_ import creat_ini, read_ini, update_ini
from sklearn.externals.joblib import load #读取 pkl 文件
# from utils.csv_transform import OneHotEncoder_, LabelEncoder_



class NN(NN_Config):
    def __init__(self):
        super(NN, self).__init__()
        self.layers = []
        self.num_samples = {
        "train": 150 ,
        "validation": 12 * 2
    }


    def creat_ini_config(self , layers , config_name="神经网络参数.ini"):
        sections = ["分类还是回归", "神经网络结构", "迭代", "学习率", "L2正则化" , "dropout" , "滑动平均"]
        options = [
            dict(classifier_or_regression=str(self.classifier_or_regression)) ,
            dict(layers = str(layers)) ,
            dict(iteration = str(self.iteration) , batch_size = str(self.batch_size)) ,
            dict(
                learning_rate=str(self.learning_rate), enable_learning_rate_decay=str(self.enable_learning_rate_decay) ,
                decay_rate = str(self.decay_rate) , decay_steps = str(self.decay_steps) ,
                staircase = str(self.staircase)
            ) ,
            dict(enable_L2_regularizer = str(self.enable_L2_regularizer) , L2_regularizer_rate = str(self.L2_regularizer_rate)) ,
            dict(enable_dropout = str(self.enable_dropout) , keep_prob = str(self.keep_prob)) ,
            dict(moving_avg_decay_rate = str(self.moving_avg_decay_rate))
        ]
        model_name = self.model_path.split("\\")[-1]
        config_driectory = self.model_path.split(model_name)[0][0:-1]
        config_Path = config_driectory + "\\" + config_name
        is_exists = os.path.exists(config_Path)
        if is_exists is False:
            creat_ini(config_Path , sections , options , encoding="utf-8")
            print("成功创建：" + config_name)
        else:
            update_ini(config_Path , sections , options , encoding="utf-8")
            print("成功更新：" + config_name)



    def read_ini_config(self , model_path , config_name="神经网络参数.ini"):
        model_name = model_path.split("\\")[-1]
        config_driectory = model_path.split(model_name)[0][0:-1]
        config_Path = config_driectory + "\\" + config_name
        sections , options = read_ini(config_Path , encoding="utf-8")
        for i in range(len(sections)):
            group = sections[i]
            option = options[i]
            if group == "分类还是回归":
                self.classifier_or_regression = option["classifier_or_regression"]
            elif group == "神经网络结构":
                layers_str = option["layers"]
                replace_Space = layers_str.replace(" " , "")
                strip_boundary = replace_Space.strip("[]")
                split_comma = strip_boundary.split(",")
                self.layers = list(map(int , split_comma))
            elif group == "迭代":
                iteration_str = option["iteration"]
                batch_size_str = option["batch_size"]
                self.iteration = int(iteration_str)
                self.batch_size = int(batch_size_str)
            elif group == "学习率":
                learning_rate_str = option["learning_rate"]
                enable_learning_rate_decay_str = option["enable_learning_rate_decay"]
                decay_rate_str = option["decay_rate"]
                decay_steps_str = option["decay_steps"]
                staircase_str = option["staircase"]
                self.learning_rate = float(learning_rate_str)
                self.enable_learning_rate_decay = bool(enable_learning_rate_decay_str)
                self.decay_rate = float(decay_rate_str)
                if decay_steps_str == "None":
                    self.decay_steps = None
                else:
                    self.decay_steps = int(decay_steps_str)
                self.staircase = bool(staircase_str)
            elif group == "L2正则化".lower():
                enable_l2_regularizer_str = option["enable_l2_regularizer".lower()]
                l2_regularizer_rate_str = option["l2_regularizer_rate".lower()]
                self.enable_l2_regularizer = bool(enable_l2_regularizer_str)
                self.l2_regularizer_rate = float(l2_regularizer_rate_str)
            elif group == "dropout":
                enable_dropout_str = option["enable_dropout"]
                keep_prob_str = option["keep_prob"]
                self.enable_dropout = bool(enable_dropout_str)
                self.keep_prob = float(keep_prob_str)
            elif group == "滑动平均":
                moving_avg_decay_rate_str = option["moving_avg_decay_rate"]
                self.moving_avg_decay_rate = float(moving_avg_decay_rate_str)




    def get_layers(self , X_train , Y_train):
        """
        功能：获取神经网络的层结构
        参数：
            hidden_layer_unit：每一个隐藏层的单元数(list类型)
            X_train：训练集中的 X
            Y_train：训练集中的 Y
        返回值：神经网络的层结构(list类型)。例如：layers[0]的值表示输入层有几个神经元，layers[1]的值表示第一个隐藏层层有几个神经元
        """
        input_layer_unit = X_train.shape[1]
        output_layer_unit = Y_train.shape[1]
        layers = [input_layer_unit] + self.hidden_layer_unit + [output_layer_unit]
        return layers




    def get_weight(self , layers , stddev=0.1 , seed=42):
        '''
        功能：获取权重
        参数：
            layers：神经网络的层结构(list类型)。例如：layers[0]的值表示输入层有几个神经元，layers[1]的值表示第一个隐藏层层有几个神经元
            stddev：方差
            seed：随机种子
        返回值：神经网络的所有权重矩阵的集合(list类型)。例如：W[0]表示输入层与第一个隐藏层之间的权重矩阵，类似的还有W[1]、W[2]等等
        '''
        W = []
        for i in range(len(layers) - 1):
            input_layer_unit = layers[i]
            output_layer_unit = layers[i + 1]
            shape = [input_layer_unit , output_layer_unit]
            var_name = "W" + str(i+1)
            W_i = tf.Variable(tf.random_normal(shape , stddev=stddev , seed=seed) , name=var_name)
            W.append(W_i)
        return W



    def get_bias(self , layers):
        '''
        功能：获取 Bias
        参数：
            layers：神经网络的层结构(list类型)。例如：layers[0]的值表示输入层有几个神经元，layers[1]的值表示第一个隐藏层层有几个神经元
        返回值：神经网络的所有Bias矩阵的集合(list类型)。例如：B[0]表示输入层与第一个隐藏层之间的Bias矩阵，类似的还有B[1]、B[2]等等
        '''
        B = []
        for i in range(len(layers) - 1):
            output_layer_unit = layers[i+1]
            var_name = "B" + str(i+1)
            B_i = tf.Variable(tf.constant(0.1, shape=[output_layer_unit]), name=var_name)
            B.append(B_i)
        return B




    # 学习率衰减
    def learning_rate_decay(self , enable_learning_rate_decay , decay_rate, batch_size , epochs_per_decay, staircase, global_step):
        '''
        功能：获取使用学习率衰减后的学习率
        参数：
            enable_learning_rate_decay：是否启用学习率衰减功能
            learning_rate：开始时的学习率
            decay_rate：学习率衰减率
            staircase：衰减的方式(True表示阶梯衰减，False表示曲线衰减)
            global_step：第几次迭代
            X_train：训练数据
            decay_steps：衰减的步伐（每 n 次迭代，学习率衰减一次）。可以选值：None、整数。
                None表示完整的使用一遍训练数据所需要的迭代次数，这个迭代次数为 n，每 n 次迭代，学习率衰减一次。
                整数表示每 n 次迭代，学习率衰减一次。
            batch_size：每次迭代的 batch 数据大小
        返回值：使用学习率衰减后的学习率
        '''
        initial_learning_rate = 0.1 * batch_size / 256
        if enable_learning_rate_decay is True:
            decay_steps = (self.num_samples["train"] / batch_size) * epochs_per_decay  # 表示：衰减的步伐。每多少个 epoch，学习率衰减一次
            # 每经过 decay_steps 次迭代，学习率衰减一次。参数
            learning_rate = tf.train.exponential_decay(
                initial_learning_rate, global_step, decay_steps, decay_rate, staircase=staircase
            )
        else:
            learning_rate = initial_learning_rate
        return learning_rate



    def get_l2_regularizer_value(self, W):
        """
        功能：L2正则化之后的值
        参数：
            enable_regularizer：是否启用L2正则化
            regularizer_rate：正则化的系数(系数越大，正则化越强，越能够防止过拟合。如果太大了，可能会欠拟合)
            W：神经网络的所有权重矩阵的集合(list类型)。例如：W[0]表示输入层与第一个隐藏层之间的权重矩阵，类似的还有W[1]、W[2]等等
        返回值：L2正则化之后的值
        """
        if self.enable_L2_regularizer is False:
            L2_valuse = None
        else:
            regularizer = tf.contrib.layers.l2_regularizer(self.L2_regularizer_rate)
            L2_list = []
            for i in range(len(W)):
                L2_W_i = regularizer(W[i])
                L2_list.append(L2_W_i)
            L2_valuse = sum(L2_list)
        return L2_valuse



    def get_accuracy_rate(self , y_true , output):
        '''
        功能：获取正确率
        参数：
            y_true：真实值
            output：输出层的输出
        返回值：正确率
        '''
        y_pred = tf.argmax(output , 1)
        y_true = tf.argmax(y_true , 1)
        correct_prediction = tf.equal(y_pred , y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
        return accuracy



    def get_current_batch_data(self , current_iteration_num , X_train , Y_train):
        '''
        功能：获取当前迭代时的 batch 数据
        参数：
            current_iteration_num：当前是第几次迭代
            X_train：训练集中的 X
            Y_train：训练集中的 Y
        返回值：当前迭代时的 batch 数据
        '''
        sample_number = X_train.shape[0] #样本数量
        start_index = (current_iteration_num * self.batch_size) % sample_number #当前batch的起始index
        end_index = min(start_index + self.batch_size  , sample_number) #当前batch的结束index
        batch_x = X_train[start_index : end_index] #当前batch的起始数据
        batch_y = Y_train[start_index : end_index] #当前batch的结束数据
        return batch_x , batch_y



    def get_time_left(self , cycle , start_time , total_use_time , current_iteration_num):
        '''
        功能：获取训练总共花费的时间、剩余时间
        参数：
            start_time：训练开始前的时间
            total_use_time：训练总共花费的时间
            current_iteration_num：当前是第几次迭代
        返回值：训练总共花费的时间、剩余时间
        '''
        time_left = None
        if current_iteration_num == cycle:
            total_use_time = (datetime.datetime.now() - start_time) * (self.iteration / cycle)
        if (current_iteration_num >= cycle) and (current_iteration_num % cycle == 0):
            now_time = datetime.datetime.now()
            cumulative_use_time = now_time - start_time
            time_left = total_use_time - cumulative_use_time
            time_left = str(time_left).split(".")[0]
        return total_use_time , time_left




    def forward_propagation(self , layers , x , W , B , keep_prob):
        """
        功能：神经网络前向传播，得到输出层的结果
        参数：
            layers：神经网络的层结构(list类型)。例如：layers[0]的值表示输入层有几个神经元，layers[1]的值表示第一个隐藏层层有几个神经元
            x：神经网络的输入，是 tf.placeholder 格式，在训练的时候用 sess.run(clf , feed_dict={x:batch_x , y_true:batch_y})
            W：神经网络的所有权重矩阵的集合(list类型)。例如：W[0]表示输入层与第一个隐藏层之间的权重矩阵，类似的还有W[1]、W[2]等等
            B：神经网络的所有Bias矩阵的集合(list类型)。例如：B[0]表示输入层与第一个隐藏层之间的Bias矩阵，类似的还有B[1]、B[2]等等
        返回值：输出层的结果
        """
        input = None
        output = None
        for i in range(len(layers) - 1):
            if i == 0:
                input = x
            Wx_plus_b = tf.matmul(input , W[i]) + B[i]
            # len(layers)-2 是排除了最后一个隐藏层与输出层这一次操作
            if i != (len(layers) - 2):
                output = tf.nn.relu(Wx_plus_b)
                # 是否启用 dropout
                if self.enable_dropout is True:  #启用
                    output = tf.nn.dropout(output , keep_prob)  # dropout 操作，keep_prob 是 placeholder 格式
                input = output
            else:
                output = Wx_plus_b
        return output



    def backward_propagation(self , y_true , output , L2_valuse , global_step):
        '''
        功能：神经网络反向传播，返回待训练的分类器
        参数：
            y_true：真实值，是 tf.placeholder 格式，在训练的时候用 sess.run(clf , feed_dict={x:batch_x , y_true:batch_y})
            output：神经网络输出层的输出
            L2_valuse：L2正则化的值
        返回值：待训练的分类器
        '''
        #定义损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true , logits=output)  #交叉熵
        cross_entropy_mean = tf.reduce_mean(cross_entropy) #每一个batch的平均交叉熵
        # 是否启用 L2 正则化
        if L2_valuse is None:  # 不启用
            loss = cross_entropy_mean
        else:  # 启用
            loss = cross_entropy_mean + L2_valuse  # 最终损失函数
        # 优化损失函数
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        minimize_loss_op = optimizer.minimize(loss , global_step=global_step)
        # 滑动平均模型
        moving_avg_obj = tf.train.ExponentialMovingAverage(self.moving_avg_decay_rate , global_step)
        moving_ave_op = moving_avg_obj.apply(tf.trainable_variables())
        # train_op 是训练对象。需要训练的对象有2个：分别是 minimize_loss_op 和 moving_ave_op
        with tf.control_dependencies([minimize_loss_op , moving_ave_op]):  # 把2个训练对象放到 train_op 中。
            train_op = tf.no_op(name="train")
        return train_op



    def train(self , train_op , x , y_true , keep_prob , X_train , Y_train , X_validation , Y_validation , accuracy , global_step):
        """
        功能：训练神经网络，得到训练好的模型
            clf: 待训练的分类器
            accuracy: 正确率，在训练的时候用 sess.run(accuracy , feed_dict={x:batch_X_data , y_true:batch_Y_data})
            module_path：模型保存的路径
            global_step：tf.Variable(0, trainable=False)
        返回值：训练好的模型
        """
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=None)
            #初始化所有变量
            init = tf.global_variables_initializer()
            sess.run(init)
            start_time = datetime.datetime.now() #训练开始时间
            total_use_time = None #训练总共花费的时间
            #迭代训练神经网络
            for i in range(self.iteration):
                batch_x , batch_y = self.get_current_batch_data(i , X_train , Y_train)
                sess.run(train_op , feed_dict={x:batch_x , y_true:batch_y , keep_prob:self.keep_prob})
                #每迭代1000次，把日志打印出来
                print_cycle = 1000
                if ((i+1) % print_cycle == 0):
                    train_accuracy = sess.run(accuracy , feed_dict={x: batch_x , y_true: batch_y , keep_prob:self.keep_prob})
                    validation_acc = sess.run(accuracy, feed_dict={x: X_validation, y_true: Y_validation, keep_prob:self.keep_prob})
                    print(str(i+1) +  "   训练集："  + str(train_accuracy) +  "   验证集："  + str(validation_acc) ,end="")
                    total_use_time , time_left = self.get_time_left(print_cycle , start_time, total_use_time, i + 1)
                    if str(time_left)[0:1] == "-":
                        print("   剩余时间：" + "None")
                    else:
                        print("   剩余时间：" + str(time_left))
            # 保存模型
            self.save_module(saver, sess, global_step)



    def save_module(self , saver , sess , global_step):
        """
        功能：保存模型
        参数：
            saver：tf.train.Saver(max_to_keep=None)
            sess：tf.Session() as sess
            global_step：tf.Variable(0, trainable=False)
        返回值：没有返回值
        """
        module_name = self.model_path.split("\\")[-1]
        driectory = self.model_path.split(module_name)[0][0:-1]
        is_exists = os.path.exists(driectory)  # 判断一个目录是否存在
        if is_exists is False:
            os.makedirs(driectory)  # 创建目录
        saver.save(sess, self.model_path , global_step=global_step)




    def fit(self , X_train, Y_train, X_validation, Y_validation):
        layers = self.get_layers(X_train , Y_train)  #获取神经网络的层结构
        W = self.get_weight(layers)  #初始化权重
        B = self.get_bias(layers)  #初始化偏向
        x = tf.placeholder(tf.float32, shape=[None, layers[0]], name="input")  # 神经网络的输入
        y_true = tf.placeholder(tf.float32, shape=[None, layers[-1]], name="output")  # 神经网络的输入
        keep_prob = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
        global_step = tf.Variable(0, trainable=False)  # 当前的迭代次数
        # 获取学习率
        # learning_rate = self.learning_rate_decay(
        #     self.enable_learning_rate_decay, decay_rate=self.decay_rate, batch_size=self.batch_size,
        #     epochs_per_decay=10, staircase=True, global_step=global_step
        # )

        L2_valuse = self.get_l2_regularizer_value(W)  # L2正则化，返回 L2 的值
        output = self.forward_propagation(layers, x, W, B , keep_prob)  #前向传播，返回输出层的输出
        accuracy = self.get_accuracy_rate(y_true, output)  #获取前向传播的正确率
        train_op = self.backward_propagation(y_true, output, L2_valuse, global_step)  #反向传播，返回待训练的分类器
        self.creat_ini_config(layers)
        self.train(train_op , x, y_true , keep_prob , X_train, Y_train, X_validation, Y_validation, accuracy , global_step) #训练





    def predict(self , model_path , X_test , Y_test , enable_moving_avg=False):
        """
        功能：预测
        参数：
            Y_test：测试集的标签。如果在生产环境中，把 Y_test 设置为 None
            enable_moving_avg：是否启用滑动平均
        """
        # 新建一个计算图作为默认计算图，起一个别名叫 g
        with tf.Graph().as_default() as g:
            self.read_ini_config(model_path)
            W = self.get_weight(self.layers)  # 初始化权重
            B = self.get_bias(self.layers)  # 初始化偏向
            x = tf.placeholder(tf.float32, shape=[None, self.layers[0]], name="input")  # 神经网络的输入
            y_true = tf.placeholder(tf.float32, shape=[None, self.layers[-1]], name="output")  # 神经网络的输入
            keep_prob = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
            output = self.forward_propagation(self.layers, x, W, B , keep_prob)  # 前向传播，返回输出层的输出
            accuracy = self.get_accuracy_rate(y_true , output)  # 获取前向传播的正确率
            # 是否使用滑动平均
            if enable_moving_avg is False:  #不使用滑动平均
                saver = tf.train.Saver()  # 实例化 Saver的时候，变量重命名
            else:  #使用滑动平均
                moving_avg_obj = tf.train.ExponentialMovingAverage(self.moving_avg_decay_rate)
                moving_avg_rename_dict = moving_avg_obj.variables_to_restore()
                saver = tf.train.Saver(moving_avg_rename_dict)  # 实例化 Saver的时候，变量重命名
            # 因为保存模型时，加上了 global_step，有了很多个模型，所以在测试时要对每一个模型进行预测
            models_path_list = self.get_models_path(model_path)  #获取所有模型的路径，返回路径的集合
            y_pred = None
            y_pred_list = []  #保存了每一个模型的预测值
            for i in range(len(models_path_list)):  #对每一个模型进行预测
                # 创建会话
                with tf.Session() as sess:
                    model_path = models_path_list[i]  #第 i 个模型的路径
                    saver.restore(sess , model_path)  # 加载第 i 个模型
                    y_pred_tensor = tf.argmax(output, 1)  #把哑编码格式的output转换为labelEncoder 的格式。目前来说仍然是tensor类型
                    y_pred_num_ndarray = sess.run(y_pred_tensor , feed_dict={x: X_test , keep_prob:1}) #返回值是数字矩阵，是numpy类型
                    # y_pred_str_ndarray = self.label_inverse_transform(y_pred_num_ndarray) #标签反转：labelEncoder.inverse_transform()
                    y_pred_list.append(y_pred_num_ndarray)
                    if Y_test is not None:
                        test_accuracy = sess.run(accuracy , feed_dict={x: X_test, y_true: Y_test , keep_prob:1})
                        print(model_path.split("-")[-1] + "次迭代，测试集正确率：" + str(test_accuracy))
        if Y_test is not None:
            y_pred = y_pred_list
        else:
            y_pred = y_pred_list[-1]
        return y_pred



    def get_models_path(self , model_path):
        module_name = model_path.split("\\")[-1]
        driectory = model_path.split(module_name)[0][0:-1]
        file_list = os.listdir(driectory)  # 列出文件夹下所有的目录与文件
        file_list = sorted(file_list , reverse=False)
        models_path_list = []
        for i in range(len(file_list)):
            filePath = driectory + "\\" + file_list[i]
            is_file = os.path.isfile(filePath)
            suffix_name = (filePath.split(".")[-1] == "meta")
            if is_file and suffix_name:
                model_path_i = filePath.split(".meta")[0]
                models_path_list.append(model_path_i)
        global_step_group = lambda i: int(re.search(r'(\d+)' , i).group())  #对 list 中的每一个元素，匹配数字部分，然后按数字分组
        models_path_list.sort(key=global_step_group , reverse=False)  #对 list 中的元素按照 key 来排序，升序
        return models_path_list




    # def label_inverse_transform(self , y_pred_num_ndarray , preprocessing_pkl_name="数据预处理.pkl"):
    #     '''
    #     功能：在预测时对神经网络的输出进行"反转换" 。因为在训练的时候，对标签进行 labelEncoder 的处理，所以在预测时要 "反转换"。
    #         转换的函数是：labelEncoder.inverse_transform()
    #
    #     y_pred_num_ndarray: 神经网络的预测值
    #         前提条件：y_pred_tensor = tf.argmax(output , 1) ， y_pred_num_ndarray = sess.run(y_pred_tensor)
    #     label: 预测的标签名称
    #     model_path: 模型路径
    #     preprocessing_pkl_name: 数据预处理持久化的文件名称
    #     返回值：转换之后的预测值
    #     '''
    #     y_pred_str_ndarray = None
    #     model_name = self.model_path.split("\\")[-1]
    #     driectory = self.model_path.split(model_name)[0][0:-1]
    #     processing_pkl_Path = driectory + "\\" + preprocessing_pkl_name
    #     preprocessing_obj_dict = load(processing_pkl_Path)
    #     if isinstance(preprocessing_obj_dict["标签预处理"] , LabelEncoder_):
    #         le = preprocessing_obj_dict["标签预处理"].labelEncoder_list[0]
    #         y_pred_str_ndarray = le.inverse_transform(y_pred_num_ndarray)
    #     elif isinstance(preprocessing_obj_dict["标签预处理"] , OneHotEncoder_):
    #         le = preprocessing_obj_dict["标签预处理"].labelEncoder.labelEncoder_list[0]
    #         y_pred_str_ndarray = le.inverse_transform(y_pred_num_ndarray)
    #     return y_pred_str_ndarray


import pandas as pd
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , StandardScaler
import warnings
warnings.filterwarnings("ignore")


def read_csv(csvPath):
    file = open(csvPath)
    dataSet = pd.read_csv(file)
    file.close()
    return dataSet


trainSet = read_csv(r"F:\data\岩石分类\trainSet\data-0.deep_learning")
validationSet = read_csv(r"F:\data\岩石分类\validationSet\data-0.deep_learning")
X_train = trainSet.loc[:, trainSet.columns != "标签"]
Y_train = trainSet.loc[:, trainSet.columns == "标签"]
X_validation = validationSet.loc[:, validationSet.columns != "标签"]
Y_validation = validationSet.loc[:, validationSet.columns == "标签"]

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_validation = ss.transform(X_validation)

le = LabelEncoder()
onehot = OneHotEncoder()
Y_train = le.fit_transform(Y_train)
Y_validation = le.transform(Y_validation)
Y_train = onehot.fit_transform(Y_train.reshape(-1, 1))
Y_train = Y_train.toarray()
Y_validation = onehot.transform(Y_validation.reshape(-1, 1))
Y_validation = Y_validation.toarray()

nn = NN()
nn.fit(X_train, Y_train, X_validation, Y_validation)
