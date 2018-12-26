# -*- coding: utf-8 -*-

import os
import stat  #用于设置文件为只读
import re
import datetime
import tensorflow as tf
from sklearn.externals.joblib import load #读取 pkl 文件
from config.configparser_ import creat_ini, update_ini, read_ini
from config.rnn_config import RNN_Config
# from utils.csv_transform import OneHotEncoder_, LabelEncoder_, StandardScaler_


class RNN(RNN_Config):
    def __init__(self):
        super(RNN, self).__init__()
        self.input_layer_unit = self.each_time_step_input  # 输入层单元数


    def creat_ini_config(self , config_name="循环神经网络参数.ini"):
        sections = ["时序" , "循环神经网络结构" , "迭代" , "学习率" , "dropout"]
        options = [
            dict(time_step = str(self.time_sequence) , each_time_step_input = str(self.each_time_step_input)) ,
            dict(
                input_layer_unit = str(self.input_layer_unit) ,
                hidden_layer_units = str(self.hidden_layer_units) ,
                lstm_layers_num = str(self.lstm_layers_num) ,
                output_layer_unit = str(self.output_layer_unit)
            ) ,
            dict(iteration = str(self.iteration) , batch_size = str(self.batch_size)) ,
            dict(learning_rate=str(self.learning_rate)) ,
            dict(enable_dropout = str(self.enable_dropout) , keep_prob = str(self.keep_prob)) ,
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
        # os.chmod(config_Path , stat.S_IREAD)  #把文件设置为只读




    def read_ini_config(self , model_path , config_name="循环神经网络参数.ini"):
        model_name = model_path.split("\\")[-1]
        config_driectory = model_path.split(model_name)[0][0:-1]
        config_Path = config_driectory + "\\" + config_name
        sections , options = read_ini(config_Path , encoding="utf-8")
        for i in range(len(sections)):
            group = sections[i]
            option = options[i]
            if group == "时序":
                time_step_str = option["time_step"]
                each_time_step_input_str = option["each_time_step_input"]
                self.time_sequence = int(time_step_str)
                self.each_time_step_input = int(each_time_step_input_str)
            elif group == "循环神经网络结构":
                input_layer_unit_str = option["input_layer_unit"]
                hidden_layer_units_str = option["hidden_layer_units"]
                lstm_layers_num_str = option["lstm_layers_num"]
                output_layer_unit_str = option["output_layer_unit"]
                self.input_layer_unit = int(input_layer_unit_str)
                self.hidden_layer_units = int(hidden_layer_units_str)
                self.lstm_layers_num = int(lstm_layers_num_str)
                self.output_layer_unit = int(output_layer_unit_str)
            elif group == "迭代":
                iteration_str = option["iteration"]
                batch_size_str = option["batch_size"]
                self.iteration = int(iteration_str)
                self.batch_size = int(batch_size_str)
            elif group == "学习率":
                learning_rate_str = option["learning_rate"]
                self.learning_rate = float(learning_rate_str)
            elif group == "dropout":
                enable_dropout_str = option["enable_dropout"]
                keep_prob_str = option["keep_prob"]
                self.enable_dropout = bool(enable_dropout_str)
                self.keep_prob = float(keep_prob_str)




    def get_weight(self , stddev=0.1 , seed=1):
        W = []
        layers = [self.input_layer_unit , self.hidden_layer_units , self.output_layer_unit]
        for i in range(len(layers) - 1):
            input_layer_unit = layers[i]
            output_layer_unit = layers[i + 1]
            shape = [input_layer_unit, output_layer_unit]
            if i == 0:
                var_name = "W_intput"
            else:
                var_name = "W_output"
            W_i = tf.Variable(tf.random_normal(shape, stddev=stddev, seed=seed), name=var_name)
            W.append(W_i)
        return W



    def get_bias(self):
        '''
        功能：获取 Bias
        参数：
            layers：神经网络的层结构(list类型)。例如：layers[0]的值表示输入层有几个神经元，layers[1]的值表示第一个隐藏层层有几个神经元
        返回值：神经网络的所有Bias矩阵的集合(list类型)。例如：B[0]表示输入层与第一个隐藏层之间的Bias矩阵，类似的还有B[1]、B[2]等等
        '''
        B = []
        layers = [self.input_layer_unit , self.hidden_layer_units , self.output_layer_unit]
        for i in range(len(layers) - 1):
            output_layer_unit = layers[i+1]
            if i == 0:
                var_name = "B_intput"
            else:
                var_name = "B_output"
            B_i = tf.Variable(tf.zeros([output_layer_unit]) , name=var_name)
            B.append(B_i)
        return B



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



    def get_current_batch_data(self , current_iteration_num, X_train, Y_train):
        '''
        功能：获取当前迭代时的 batch 数据
        参数：
            current_iteration_num：当前是第几次迭代
            X_train：训练集中的 X
            Y_train：训练集中的 Y
        返回值：当前迭代时的 batch 数据
        '''
        sample_number = X_train.shape[0]  # 样本数量
        start_index = (current_iteration_num * self.batch_size) % (sample_number - self.batch_size)  # 当前batch的起始index
        end_index = start_index + self.batch_size  # 当前batch的结束index
        batch_x = X_train[start_index: end_index]  # 当前batch的起始数据
        batch_y = Y_train[start_index: end_index]  # 当前batch的结束数据
        return batch_x, batch_y



    def get_time_left(self , cycle , start_time, total_use_time, current_iteration_num):
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
        return total_use_time, time_left




    def get_accuracy_rate(self , y_true, output):
        '''
        功能：获取正确率
        参数：
            y_true：真实值
            output：输出层的输出
        返回值：正确率
        '''
        y_pred = tf.argmax(output, 1)
        y_true = tf.argmax(y_true, 1)
        correct_prediction = tf.equal(y_pred, y_true)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy




    def forward_propagation(self , x , W , B , keep_prob):
        # 输入层与隐藏层之间是全连接关系，进行 Wx_plus_b 操作
        x = tf.reshape(x , [-1 , self.each_time_step_input])  # shape = [batch_size * time_step , each_time_step_input]
        Wx_plus_b = tf.matmul(x , W[0]) + B[0]  # W*X + b
        Wx_plus_b = tf.reshape(Wx_plus_b , [-1 , self.time_sequence , self.hidden_layer_units])
        # 隐藏层 与 LSTMCell 之间是 RNN 连接关系
        lstm_input = Wx_plus_b
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_layer_units , forget_bias=1.0 , state_is_tuple=True)
        if self.enable_dropout is True:  # 是否启用 dropout。如果是 True，说明启用 dropout
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell , output_keep_prob=keep_prob)  # dropout 操作
        multi_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(self.lstm_layers_num)] , state_is_tuple=True) #多层 LSTM
        outputs , final_state = tf.nn.dynamic_rnn(multi_lstm , lstm_input , dtype=tf.float32)  # 多层 LSTM 的输出
        outputs_final_state = tf.transpose(outputs , (1, 0, 2))  # 最后一个状态的 output
        # 最后一个 LSTMCell 与 输出层之间是全连接关系，进行 Wx_plus_b 操作
        fc_input = outputs_final_state[-1] # 把最后一个状态的 output 作为全连接的输入
        output = tf.matmul(fc_input, W[1]) + B[1]
        return output




    def backward_propagation(self , y_true , output , L2_valuse ,  global_step):
        '''
        功能：神经网络反向传播，返回待训练的分类器
        参数：
            y_true：真实值，是 tf.placeholder 格式，在训练的时候用 sess.run(clf , feed_dict={x:batch_x , y_true:batch_y})
            output：神经网络输出层的输出
            L2_valuse：L2正则化的值
        返回值：待训练的分类器
        '''
        # 定义损失函数
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=output)  # 交叉熵
        cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 每一个batch的平均交叉熵
        # 是否启用 L2 正则化
        if L2_valuse is None:  # 不启用
            loss = cross_entropy_mean
        else:  # 启用
            loss = cross_entropy_mean + L2_valuse  # 最终损失函数
        # 优化损失函数
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        minimize_loss_op = optimizer.minimize(loss, global_step=global_step)
        train_op = minimize_loss_op
        return train_op




    def train(self , train_op , x , y_true , X_train , Y_train , X_validation , Y_validation , keep_prob , accuracy , global_step):
        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        mnist = read_data_sets("data" , one_hot=True)
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=None)
            # 初始化所有变量
            init = tf.global_variables_initializer()
            sess.run(init)
            start_time = datetime.datetime.now()  # 训练开始时间
            total_use_time = None  # 训练总共花费的时间
            # 迭代训练神经网络
            for i in range(self.iteration):
                batch_x , batch_y = mnist.train.next_batch(self.batch_size, shuffle=False)
                X_validation , Y_validation = mnist.validation.next_batch(self.batch_size, shuffle=False)
                # batch_x , batch_y = self.get_current_batch_data(i , X_train , Y_train)
                sess.run(train_op, feed_dict={x:batch_x , y_true:batch_y , keep_prob:self.keep_prob})
                print_cycle = 100
                if (i + 1) % print_cycle == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={x:batch_x , y_true:batch_y , keep_prob:self.keep_prob})
                    validation_acc = sess.run(accuracy, feed_dict={x: X_validation, y_true: Y_validation, keep_prob:self.keep_prob})
                    print(str(i + 1) + "   训练集：" + str(train_accuracy) + "   验证集" + str(validation_acc), end="")
                    total_use_time, time_left = self.get_time_left(print_cycle , start_time, total_use_time, i + 1)
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
        W = self.get_weight()  #初始化权重
        B = self.get_bias()  #初始化偏向
        x = tf.placeholder(tf.float32, [None , 784])
        # x = tf.placeholder(tf.float32, [None , X_train.shape[1]])
        y_true = tf.placeholder(tf.float32, [None , self.output_layer_unit])
        keep_prob = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
        global_step = tf.Variable(0, trainable=False)  # 当前的迭代次数
        L2_valuse = self.get_l2_regularizer_value(W)  # L2正则化，返回 L2 的值
        output = self.forward_propagation(x , W , B , keep_prob)  #前向传播，返回输出层的输出
        accuracy = self.get_accuracy_rate(y_true , output)  #获取前向传播的正确率
        train_op = self.backward_propagation(y_true , output , L2_valuse , global_step)  #反向传播，返回待训练的分类器
        self.creat_ini_config()
        self.train(train_op , x , y_true , X_train , Y_train , X_validation , Y_validation ,keep_prob , accuracy, global_step) #训练



    def predict(self , model_path , X_test , Y_test):
        """
        功能：预测
        参数：
            Y_test：测试集的标签。如果在生产环境中，把 Y_test 设置为 None
            enable_moving_avg：是否启用滑动平均
        """
        # 新建一个计算图作为默认计算图，起一个别名叫 g
        with tf.Graph().as_default() as g:
            self.read_ini_config(model_path)
            W = self.get_weight()  # 初始化权重
            B = self.get_bias()  # 初始化偏向
            # 循环神经网络的输入
            x = tf.placeholder(tf.float32, [None, 784])
            # x = tf.placeholder(tf.float32, [None , X_test.shape[1]])
            y_true = tf.placeholder(tf.float32, [None , self.output_layer_unit])
            keep_prob = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
            output = self.forward_propagation(x , W , B , keep_prob)  # 前向传播，返回输出层的输出
            accuracy = self.get_accuracy_rate(y_true , output)  # 获取前向传播的正确率
            saver = tf.train.Saver()  # 实例化 Saver的时候，变量重命名
            # 因为保存模型时，加上了 global_step，有了很多个模型，所以在测试时要对每一个模型进行预测
            models_path_list = self.get_models_path(model_path)  # 获取所有模型的路径，返回路径的集合
            y_pred = None
            y_pred_list = []  # 保存了每一个模型的预测值
            for i in range(len(models_path_list)):  # 对每一个模型进行预测
                # 创建会话
                with tf.Session() as sess:
                    model_path = models_path_list[i]  # 第 i 个模型的路径
                    saver.restore(sess, model_path)  # 加载第 i 个模型
                    y_pred_tensor = tf.argmax(output , 1) #把哑编码格式的output转换为 labelEncoder 的格式。目前来说仍然是 tensor 类型
                    y_pred_num_ndarray = sess.run(y_pred_tensor ,feed_dict={x: X_test, keep_prob: 1}) #返回值是数字矩阵，是numpy类型
                    # y_pred_str_ndarray = self.label_inverse_transform(y_pred_num_ndarray) #标签反转：labelEncoder.inverse_transform()
                    y_pred_list.append(y_pred_num_ndarray)
                    if Y_test is not None:
                        test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_true: Y_test, keep_prob: 1})
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
    #     elif isinstance(preprocessing_obj_dict["标签预处理"] , StandardScaler_):
    #         le = preprocessing_obj_dict["标签预处理"].standardScaler
    #         y_pred_str_ndarray = le.inverse_transform(y_pred_num_ndarray)
    #     return y_pred_str_ndarray


rnn = RNN()
rnn.fit(X_train=None, Y_train=None, X_validation=None, Y_validation=None)