# -*- coding: utf-8 -*-

import os
import stat  #用于设置文件为只读
import re
import datetime
import tensorflow as tf
from config.cnn_config import CNN_Config
from config.configparser_ import creat_ini, update_ini, read_ini


class CNN(CNN_Config):
    def __init__(self):
        super(CNN, self).__init__()



    def creat_ini_config(self , config_name="卷积神经网络参数.ini"):
        sections = ["图片大小" , "卷积神经网络结构", "迭代", "学习率", "L2正则化" , "dropout"]
        options = [
            dict(image_size = str(self.image_size)) ,
            dict(layers = str(self.layers)) ,
            dict(iteration = str(self.iteration) , batch_size = str(self.batch_size)) ,
            dict(learning_rate=str(self.learning_rate)) ,
            dict(enable_L2_regularizer = str(self.enable_L2_regularizer) , L2_regularizer_rate = str(self.L2_regularizer_rate)) ,
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



    def read_ini_config(self , model_path , config_name="卷积神经网络参数.ini"):
        model_name = model_path.split("\\")[-1]
        config_driectory = model_path.split(model_name)[0][0:-1]
        config_Path = config_driectory + "\\" + config_name
        sections , options = read_ini(config_Path , encoding="utf-8")
        for i in range(len(sections)):
            group = sections[i]
            option = options[i]
            if group == "图片大小":
                image_size_str = option["image_size"]
                strip_boundary = image_size_str.strip("[]")
                split_comma = strip_boundary.split(",")
                image_size = list(map(int , split_comma))
                self.image_size = image_size
            elif group == "卷积神经网络结构":
                layers_str = option["layers"]
                replace_Space = layers_str.replace(" " , "")
                strip_boundary = replace_Space.strip("[]")
                split_comma = strip_boundary.split("},{")
                layers_list = []
                for j in range(len(split_comma)):
                    key = split_comma[j].split(":")[0][1:].strip("\'\'")
                    value = split_comma[j].split(":")[1].strip("[]}")
                    value = value.split(",")
                    value = list(map(int , value))
                    layers_list.append({key:value})
                self.layers = layers_list
            elif group == "迭代":
                iteration_str = option["iteration"]
                batch_size_str = option["batch_size"]
                self.iteration = int(iteration_str)
                self.batch_size = int(batch_size_str)
            elif group == "学习率":
                learning_rate_str = option["learning_rate"]
                self.learning_rate = float(learning_rate_str)
            elif group == "L2正则化".lower():
                enable_l2_regularizer_str = option["enable_l2_regularizer".lower()]
                l2_regularizer_rate_str = option["l2_regularizer_rate".lower()]
                self.enable_L2_regularizer = bool(enable_l2_regularizer_str)
                self.L2_regularizer_rate = float(l2_regularizer_rate_str)
            elif group == "dropout":
                enable_dropout_str = option["enable_dropout"]
                keep_prob_str = option["keep_prob"]
                self.enable_dropout = bool(enable_dropout_str)
                self.keep_prob = float(keep_prob_str)




    def get_conv_layers_params(self , stddev=0.1 , seed=1):
        conv_layers = []
        # 从 继承的父类中获取卷积层的数据，但是信息还不完整
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i + 1]
            for key in current_layer:
                if key == "卷积":
                    conv_layers.append(current_layer[key])
        # 完整的构造出卷积层的信息
        for i in range(len(conv_layers)):
            conv_layers[i].insert(0 , conv_layers[i][0])
            if i == 0:
                conv_layers[i].insert(2 , self.image_size[-1])
            else:
                conv_layers[i].insert(2 , conv_layers[i-1][3])
        # 初始化权重和偏向
        W_conv = []
        B_conv = []
        for i in range(len(conv_layers)):
            with tf.name_scope("conv" + str(i + 1)):
                W_shape = conv_layers[i][0:-1]
                B_shape = conv_layers[i][-2]
                W_name = "W_conv" + str(i + 1)
                B_name = "B_conv" + str(i + 1)
                W_conv_i = tf.Variable(tf.random_normal(W_shape , stddev=stddev, seed=seed), name=W_name)
                B_conv_i = tf.Variable(tf.constant(0.1, shape=[B_shape]), name=B_name)
                W_conv.append(W_conv_i)
                B_conv.append(B_conv_i)
                tf.summary.histogram("B_conv", B_conv_i)  # 把 B_conv_i 加到 tensorboard（summary 中）
                tf.summary.histogram("W_conv" , W_conv_i)  # 把 W_conv_i 加到 tensorboard（summary 中）
        return W_conv , B_conv



    def get_fc_layers_params(self , fc_layer_input_unit , stddev=0.1 , seed=1):
        fc_layers = [fc_layer_input_unit]
        for i in range(len(self.layers) - 1):
            current_layer = self.layers[i + 1]
            for key in current_layer:
                if (key == "全连接") or (key == "输出"):
                    if type(current_layer[key]) is int:
                        current_layer[key] = [current_layer[key]]
                    fc_layers = fc_layers + current_layer[key]
        W_fc = []
        B_fc = []
        for i in range(len(fc_layers) - 1):
            with tf.name_scope("fc" + str(i + 1)):
                input_layer_unit = fc_layers[i]
                output_layer_unit = fc_layers[i + 1]
                W_shape = [input_layer_unit , output_layer_unit]
                B_shape = [output_layer_unit]
                W_name = "W_fc" + str(i + 1)
                B_name = "B_fc" + str(i + 1)
                W_fc_i = tf.Variable(tf.random_normal(W_shape , stddev=stddev, seed=seed), name=W_name)
                B_fc_i = tf.Variable(tf.zeros([output_layer_unit]), name=B_name)
                W_fc.append(W_fc_i)
                B_fc.append(B_fc_i)
                tf.summary.histogram("B_fc", B_fc_i)  # 把 B_fc_i 加到 tensorboard（summary 中）
                tf.summary.histogram("W_fc", W_fc_i)  # 把 W_fc_i 加到 tensorboard（summary 中）
        return W_fc , B_fc



    def get_l2_regularizer_value(self, W):
        """
        功能：L2正则化之后的值
        参数：
            enable_regularizer：是否启用L2正则化
            regularizer_rate：正则化的系数(系数越大，正则化越强，越能够防止过拟合。如果太大了，可能会欠拟合)
            W：神经网络的所有权重矩阵的集合(list类型)。例如：W[0]表示输入层与第一个隐藏层之间的权重矩阵，类似的还有W[1]、W[2]等等
        返回值：L2正则化之后的值
        """
        with tf.name_scope("l2_regularizer"):
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
        """
        功能：获取正确率
        参数：
            y_true：真实值
            output：输出层的输出
        返回值：正确率
        """
        with tf.name_scope("accuracy"):
            y_pred = tf.argmax(output , 1)
            y_true = tf.argmax(y_true , 1)
            correct_prediction = tf.equal(y_pred , y_true)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32))
            tf.summary.scalar("accuracy" , accuracy)  # 把 accuracy 值加到 tensorboard 中
        return accuracy




    def get_current_batch_data(self , current_iteration_num , X_train , Y_train):
        """
        功能：获取当前迭代时的 batch 数据
        参数：
            current_iteration_num：当前是第几次迭代
            X_train：训练集中的 X
            Y_train：训练集中的 Y
        返回值：当前迭代时的 batch 数据
        """
        sample_number = X_train.shape[0] #样本数量
        start_index = (current_iteration_num * self.batch_size) % sample_number #当前batch的起始index
        end_index = min(start_index + self.batch_size  , sample_number) #当前batch的结束index
        batch_x = X_train[start_index : end_index] #当前batch的起始数据
        batch_y = Y_train[start_index : end_index] #当前batch的结束数据
        return batch_x , batch_y



    def get_time_left(self , cycle , start_time , total_use_time , current_iteration_num):
        """
        功能：获取训练总共花费的时间、剩余时间
        参数：
            start_time：训练开始前的时间
            total_use_time：训练总共花费的时间
            current_iteration_num：当前是第几次迭代
        返回值：训练总共花费的时间、剩余时间
        """
        time_left = None
        if current_iteration_num == cycle:
            total_use_time = (datetime.datetime.now() - start_time) * (self.iteration / cycle)
        if (current_iteration_num >= cycle) and (current_iteration_num % cycle == 0):
            now_time = datetime.datetime.now()
            cumulative_use_time = now_time - start_time
            time_left = total_use_time - cumulative_use_time
            time_left = str(time_left).split(".")[0]
        return total_use_time , time_left



    def forward_propagation(self , x_image ,  W_conv , B_conv , keep_prob):
        input = None
        output = None
        conv_num = 0
        W_fc = None
        for i in range(len(self.layers) - 1):
            if i == 0:
                input = x_image
            current_layer = self.layers[i + 1]
            layer_name = ""
            for key in current_layer.keys():
                layer_name = key
            if layer_name == "卷积":
                with tf.name_scope("conv" + str(conv_num + 1) + "/"):
                    strides = [1 , current_layer[layer_name][-1] , current_layer[layer_name][-1] , 1]
                    Wx = tf.nn.conv2d(input=input , filter=W_conv[conv_num], strides=strides , padding="SAME")  # W * x
                    Wx_plus_b = tf.nn.bias_add(Wx , B_conv[conv_num])  # W * x + B
                    output = tf.nn.relu(Wx_plus_b)  # relu 激活函数
                    tf.summary.histogram("relu_conv" , output)
                    input = output
                    conv_num = conv_num + 1
            if layer_name == "池化":
                with tf.name_scope("pool" + str(conv_num)):
                    ksize = [1 , current_layer[layer_name][0] , current_layer[layer_name][0] , 1]
                    strides = [1 , current_layer[layer_name][-1] , current_layer[layer_name][-1] , 1]
                    output = tf.nn.max_pool(input , ksize=ksize , strides=strides , padding="SAME")
                    input = output
            if layer_name == "全连接":
                output , W_fc = self.fc_layers_forward_propagation(input , keep_prob)
        return output , W_fc



    def fc_layers_forward_propagation(self , input , keep_prob):
        fc_num = 0
        with tf.name_scope("fc" + str(1) + "/"):
            # 由池化层转入全连接层的时候，先获取池化层的输出，然后得到全连接层的输入，然后获取全连接层的权重和偏向
            pool_shape = input.get_shape().as_list()
            fc_layer_input_unit = pool_shape[1] * pool_shape[2] * pool_shape[3]  # 全连接的输入层神经元个数
            # 得到全连接层的输入。把最后一个池化层进行 reshape() 操作，shape=[batch_size , 宽*高*深度]
            input = tf.reshape(input , shape=[-1, fc_layer_input_unit])
        # 获取全连接的权重和偏向
        W_fc , B_fc = self.get_fc_layers_params(fc_layer_input_unit)
        # 前向传播
        output = None
        for j in range(len(W_fc)):
            with tf.name_scope("fc" + str(j + 1) + "/"):
                Wx_plus_b = tf.matmul(input , W_fc[fc_num]) + B_fc[fc_num]
                fc_num = fc_num + 1
                # len(layers)-1 是排除了最后一个隐藏层与输出层这一次操作
                if j != (len(W_fc) - 1):
                    output = tf.nn.relu(Wx_plus_b)
                    tf.summary.histogram("relu_fc", output)
                    # 是否启用 dropout
                    if self.enable_dropout is True:  # 启用
                        output = tf.nn.dropout(output, keep_prob)  # dropout 操作，keep_prob 是 placeholder 格式
                    input = output
                else:
                    output = Wx_plus_b
        return output , W_fc




    def backward_propagation(self , y_true ,output , L2_valuse , global_step):
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true , logits=output)  #交叉熵
            cross_entropy_mean = tf.reduce_mean(cross_entropy) #每一个batch的平均交叉熵
            # 是否启用 L2 正则化
            if L2_valuse is None:  # 不启用
                loss = cross_entropy_mean
            else:  # 启用
                loss = cross_entropy_mean + L2_valuse  # 最终损失函数
            tf.summary.scalar("loss" , loss)  # 把 loss 值加到 tensorboard 中
        # 优化损失函数
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss , global_step=global_step)
        return train_op



    def train(self , train_op , x , y_true , X_train , Y_train , X_validation , Y_validation , keep_prob ,  accuracy , global_step):
        from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
        mnist = read_data_sets("data" , one_hot=True)
        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=None)
            #初始化所有变量
            init = tf.global_variables_initializer()
            sess.run(init)
            # 保存计算图到 Tensorboard 中
            writer_summary = self.save_graph_to_tensorboard(sess)
            tensorboard_summary = tf.summary.merge_all()  # 把代码中全部 tf.summary("某命名空间" , value) 加到 tensorboard_summary 中
            # 训练时间
            start_time = datetime.datetime.now()  # 训练开始时间
            total_use_time = None  # 训练总共花费的时间
            #迭代训练神经网络
            for i in range(self.iteration):
                batch_x , batch_y = mnist.train.next_batch(self.batch_size)
                X_validation , Y_validation = mnist.validation.next_batch(self.batch_size)
                sess.run(train_op , feed_dict={x: batch_x, y_true: batch_y, keep_prob:self.keep_prob})
                print_cycle = 5
                if (i + 1) % print_cycle == 0:
                    train_feed_dict = {x: batch_x, y_true: batch_y, keep_prob:self.keep_prob}
                    validation_feed_dict = {x: X_validation, y_true: Y_validation, keep_prob: self.keep_prob}
                    train_accuracy = sess.run(accuracy , feed_dict=train_feed_dict)
                    validation_acc , summary = sess.run([accuracy , tensorboard_summary] , feed_dict=validation_feed_dict)
                    writer_summary.add_summary(summary , i+1)  # 每 i+1 次 把 summ 加到 tensorboard 中
                    print(str(i + 1) + "   训练集：" + str(train_accuracy) + "   验证集：" + str(validation_acc) , end="")
                    total_use_time, time_left = self.get_time_left(print_cycle , start_time, total_use_time , i + 1)
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
        saver.save(sess, self.model_path , global_step=global_step)  # 模型保存的核心代码



    def save_graph_to_tensorboard(self, sess):
        module_name = self.model_path.split("\\")[-1]
        driectory = self.model_path.split(module_name)[0][0:-1]
        is_exists = os.path.exists(driectory)  # 判断一个目录是否存在
        if is_exists is False:
            os.makedirs(driectory)  # 创建目录
        tensorboard_driectory = driectory + r"\tensorboard"
        writer_summary = tf.summary.FileWriter(tensorboard_driectory , sess.graph)  # 写入 tensorboard 的核心代码
        #把 write_data 中的内容写入 .bat 文件
        bat_filePath = tensorboard_driectory + r"\启动tensorboard.bat"
        write_data = "@echo off\ncd ..\nset pard=%cd%\npopd\ncd %pard%\ntensorboard --logdir=tensorboard"
        with open(bat_filePath , "w") as file:  # 第一个参数是文件路径。第二个参数是"w"，代表写入。最后赋值给file对象
            file.write(write_data)  # file对象调用write()函数，把变量a中的字符串写入文件
        return writer_summary



    def fit(self , X_train , Y_train , X_validation , Y_validation):
        self.creat_ini_config()
        imgae_width = self.image_size[0]
        imgae_high = self.image_size[1]
        color_channel = self.image_size[2]
        W_conv , B_conv = self.get_conv_layers_params()
        # 卷积神经网络的输入
        with tf.name_scope("input"):
            with tf.name_scope("x"):
                x = tf.placeholder(tf.float32 , shape=[None , self.layers[0]["输入"][0]] , name="x_numpy")  # 神经网络的输入
                x_image = tf.reshape(x , shape=[-1 , imgae_width , imgae_high , color_channel] , name="x_image")
                tf.summary.image("x_image" , x_image , 3)  # 把 x_image 中的3张图片保存到 tensorboard（summary 中）
            y_true = tf.placeholder(tf.float32, shape=[None, self.layers[-1]["输出"][0]], name="y_true")  # 神经网络的输入
        keep_prob = tf.placeholder(tf.float32 , name="keep_prob")  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
        output , W_fc = self.forward_propagation(x_image ,  W_conv , B_conv , keep_prob)
        accuracy = self.get_accuracy_rate(y_true , output)
        L2_valuse = self.get_l2_regularizer_value(W_fc)  # L2正则化，返回 L2 的值
        global_step = tf.Variable(0, trainable=False , name="global_step")  # 当前的迭代次数
        train_op = self.backward_propagation(y_true ,output , L2_valuse , global_step)  # 反向传播，返回待训练的分类器
        self.train(train_op , x , y_true , X_train , Y_train , X_validation , Y_validation , keep_prob ,  accuracy , global_step)




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
            imgae_width = self.image_size[0]
            imgae_high = self.image_size[1]
            color_channel = self.image_size[2]
            W_conv , B_conv = self.get_conv_layers_params()
            # 卷积神经网络的输入
            x = tf.placeholder(tf.float32 , shape=[None , self.layers[0]["输入"][0]] , name="input_data")  # 神经网络的输入
            y_true = tf.placeholder(tf.float32 , shape=[None , self.layers[-1]["输出"][0]] , name="output")  # 神经网络的输入
            x_image = tf.reshape(x , shape=[-1 , imgae_width, imgae_high, color_channel] , name="input_image")
            keep_prob = tf.placeholder(tf.float32)  # 不被dropout的数据比例。在sess.run()时设置keep_prob具体的值
            output , W_fc = self.forward_propagation(x_image , W_conv, B_conv, keep_prob)
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


