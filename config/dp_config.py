# -*- coding: utf-8 -*-


class DP_Config(object):
    def __init__(self):
        # deep_learning 路径
        self.csv_path_regular_expression = r"C:\Users\lenovo\Desktop\新建文件夹\原始数据\鸢尾花-*.deep_learning"

        # 分批读取 deep_learning
        self.chunksize = 100  # 每个批次的大小

        # 分割训练集、测试集、验证集
        self.test_size = 0.2
        self.validation_size = 0.1

        # 标签名称
        self.label_columns = ["标签"]

        # 离散化，然后哑编码
        # discretization_columns 是离散化的列名 list 集合。bins 是 list of list 形式，要与 discretization_columns 一一对应
        self.discretization_columns = None
        self.bins = [[4.3 , 5.3 , 6.3] , [4.3 , 5.3 , 6.3]]

        # 哑编码
        self.OneHot_columns = ["标签"]

        # 标准化
        self.StandardScaler_columns = ["花瓣长" , "花瓣宽" , "花萼长" , "花萼宽"]

