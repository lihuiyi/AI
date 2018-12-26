# -*- coding: utf-8 -*-

# import collections

class CNN_Config(object):
    def __init__(self):
        '''
        卷积层：[宽高 , 输出深度 , 步长]。  宽高一般取值：3*3 或者 5*5
        池化层：[宽高 , 步长]。  宽高一般取值：2*2 或者 3*3
        全连接层：[第1个隐藏层神经元个数 , 第2个隐藏层神经元个数 , 第3个隐藏层神经元个数 ,  ...........]
        '''
        # 模型路径
        self.model_path = r"C:\Users\lenovo\Desktop\CNN_手写数字识别模型\CNN_手写数字识别模型.ckpt"

        # 图片的大小：[宽 , 高 , 颜色通道]
        self.image_size = [28, 28, 1]

        # 卷积神经网络的结构
        self.layers = [
            dict(输入 = [784]) ,
            dict(卷积 = [3 , 64 , 1]) ,
            dict(池化 = [2 , 1]) ,
            dict(卷积 = [3 , 128 , 1]) ,
            dict(池化 = [2 , 1]) ,
            dict(全连接 = [512]) ,
            dict(输出 = [10])
        ]

        # 迭代
        self.iteration = 1000  # 总共迭代次数
        self.batch_size = 32  # 每次迭代的 batch 大小

        # 学习率
        self.learning_rate = 0.001  # 学习率。可选值：0.00001、0.0001、0.001、0.003、0.01、0.03、0.1、0.3、1、3、10

        # L2正则化
        self.enable_L2_regularizer = True  # 是否启用 L2正则化
        self.L2_regularizer_rate = 0.01  # L2 正则化系数。可选值：0.001、0.003、0.01、0.03、0.1、0.3、1、3、10

        # dropout
        self.enable_dropout = True  # 是否启用 dropout
        self.keep_prob = 0.5  # 保留神经元的比例

