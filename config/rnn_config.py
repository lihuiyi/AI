# -*- coding: utf-8 -*-

class RNN_Config(object):
    def __init__(self):
        # 模型路径
        self.model_path = r"C:\Users\lenovo\Desktop\航班\model.ckpt"

        # 时序
        self.time_sequence = 10
        self.each_time_step_input = 1

        # 循环神经网络结构
        self.hidden_layer_units = 128  # 隐藏层单元数
        self.lstm_layers_num = 2  # LSTM 层的层数
        self.output_layer_unit = 10  # 输出层单元数


        # 迭代
        self.iteration = 3000  # 总共迭代次数
        self.batch_size = 32  # 每次迭代的 batch 大小

        # 学习率
        self.learning_rate = 0.0001   # 学习率。可选值：0.00001、0.0001、0.001、0.003、0.01、0.03、0.1、0.3、1、3、10

        # L2正则化
        self.enable_L2_regularizer = False  # 是否启用 L2正则化
        self.L2_regularizer_rate = 0.01  # L2 正则化系数。可选值：0.001、0.003、0.01、0.03、0.1、0.3、1、3、10

        # dropout
        self.enable_dropout = True   # 是否启用 dropout
        self.keep_prob = 0.5  # 保留神经元的比例

