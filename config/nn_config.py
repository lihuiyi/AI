# -*- coding: utf-8 -*-


class NN_Config(object):
    def __init__(self):
        # 分类还是回归。有2个可选值："分类"、"回归"
        self.classifier_or_regression = "分类"

        # 模型路径
        self.model_path = r"F:\model\岩石分类\岩石分类.ckpt"

        # 隐藏层结构
        self.hidden_layer_unit = [121 , 61]  # 隐藏层单元数

        # 迭代
        self.iteration = 100000  # 总共迭代次数
        self.batch_size = 64  # 每次迭代的 batch 大小

        # 学习率
        self.learning_rate = 0.0001  # 学习率。可选值：0.00001、0.0001、0.001、0.003、0.01、0.03、0.1、0.3、1、3、10
        self.enable_learning_rate_decay = False  # 是否启用学习率率衰
        self.decay_rate = 0.95  # 衰减率
        self.decay_steps = None  # 衰减的步伐。可以选值：None、整数。
            # None：表示完整的使用一遍训练数据所需要的迭代次数，这个迭代次数叫做 n ，每 n 次迭代，学习率衰减一次
            # 整数：表示每 n 次迭代，学习率衰减一次
        self.staircase = True  # 衰减的方式。True表示阶梯衰减，False表示曲线衰减

        # L2正则化
        self.enable_L2_regularizer = True  # 是否启用 L2正则化
        self.L2_regularizer_rate = 0.01  # L2 正则化系数。可选值：0.001、0.003、0.01、0.03、0.1、0.3、1、3、10

        # dropout
        self.enable_dropout = True  # 是否启用 dropout
        self.keep_prob = 0.5  # 保留神经元的比例

        # 滑动平均
        self.moving_avg_decay_rate = 0.9999
