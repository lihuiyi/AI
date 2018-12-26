# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split  # 分割训练集与测试集
from sklearn.preprocessing import StandardScaler , LabelEncoder
import pickle
import warnings
warnings.filterwarnings("ignore")



class Data_Split():
    """
    功能：
        分割训练集、验证集、测试集。
        然后把数据预处理的规则保存到 pkl 文件，还没有进行预处理操作。
    """

    def __init__(self, classifier_or_regressor, label_name, label_preprocessing_method,
                 features_discretization_columns, discretization_bins, features_onehot_columns, features_standardScaler_columns):
        # 分类还是回归
        self.classifier_or_regressor = classifier_or_regressor
        # 列
        self.columns = None
        self.record_defaults = None
        # 标签预处理
        self.label_name = label_name
        self.label_preprocessing_method = label_preprocessing_method  # "标准化" 或者 None
        self.label_standardScaler_obj = StandardScaler()
        self.label_unique_value_list = []
        # 特征离散化
        self.features_discretization_columns = features_discretization_columns
        self.discretization_bins = discretization_bins
        # 特征哑编码
        self.features_onehot_columns = features_onehot_columns
        self.features_unique_value_list = []
        # 特征标准化
        self.features_standardScaler_columns = features_standardScaler_columns
        self.features_standardScaler_obj = StandardScaler()



    def get_pkl_obj(self):
        pkl_obj = dict(
            # 分类还是回归
            classifier_or_regressor = self.classifier_or_regressor,
            # 列
            columns = self.columns ,
            record_defaults = self.record_defaults ,
            # 标签预处理
            label_name = self.label_name ,
            label_preprocessing_method = self.label_preprocessing_method ,
            label_standardScaler_obj = self.label_standardScaler_obj ,
            label_unique_value_list = self.label_unique_value_list,
            # 特征离散化
            features_discretization_columns = self.features_discretization_columns ,
            discretization_bins = self.discretization_bins ,
            # 特征哑编码
            features_onehot_columns = self.features_onehot_columns ,
            features_unique_value_list = self.features_unique_value_list ,
            # 特征标准化
            features_standardScaler_columns = self.features_standardScaler_columns ,
            features_standardScaler_obj = self.features_standardScaler_obj ,
        )
        return pkl_obj



    def create_dir(self, original_csv_path, dir_name="train_data"):
        root_dir = os.path.dirname(original_csv_path)  # 例如 r"F:\岩石分类"
        data_dir = os.path.join(root_dir, dir_name)
        is_exists = os.path.exists(data_dir)  # 判断一个目录是否存在
        if is_exists is False:
            os.makedirs(data_dir)  # 创建目录
        return data_dir



    def train_test_eval_split(self, chunk_data, test_size, eval_size):
        # 在分割数据之前，先洗牌
        chunk_data = shuffle(chunk_data, random_state=42)
        chunk_data = chunk_data.reset_index(drop=True)
        # 下面开始分割
        # 根据 classifier_or_regressor 是"分类"还是"回归"，来确定 stratify 的取值
        if (self.classifier_or_regressor == "分类") or (self.classifier_or_regressor == "classifier"):
            stratify = chunk_data[self.label_name]
        else:
            stratify = None
        # 分割训练集和测试集
        train_data, test_data = train_test_split(chunk_data, test_size=test_size, shuffle=True, random_state=42, stratify=stratify)
        # 解决训练集下标乱序的问题
        train_data = train_data.reset_index(drop=True)
        # 根据 classifier_or_regressor 是"分类"还是"回归"，来确定 stratify 的取值
        if (self.classifier_or_regressor == "分类") or (self.classifier_or_regressor == "classifier"):
            stratify = train_data[self.label_name]
        else:
            stratify = None
        # 分割训练集和验证集。random_state 取值为 42 或者 84
        train_data, eval_data = train_test_split(train_data, test_size=eval_size, shuffle=True, random_state=42, stratify=stratify)
        return train_data, eval_data, test_data



    def get_record_defaults(self , data):
        record_defaults = []
        top_n_data_dtypes = data.dtypes
        # 构造出解码是需要用到的 record_defaults参数
        for dtype in top_n_data_dtypes:
            if dtype == "float64":
                record_defaults.append([0.])
            elif dtype == "int64":
                record_defaults.append([0])
            elif dtype == "object":
                record_defaults.append([""])
        return record_defaults



    def preprocessing(self, current_iteration_num, train_data):
        # 获取全部列的名称，获取 record_defaults
        if self.columns is None:
            self.columns = train_data.columns.tolist()
            self.record_defaults = self.get_record_defaults(train_data)
        # 标签预处理。分类和回归的处理方式不同
        if (self.classifier_or_regressor == "分类") or (self.classifier_or_regressor == "classifier"):
            self.unique_value(current_iteration_num, train_data, self.label_name, self.label_unique_value_list)
            self.label_standardScaler_obj = None
        else:
            if self.label_preprocessing_method == "标准化":
                self.label_standardScaler_obj.partial_fit(train_data[self.label_name].reshape(-1, 1))
            else:
                self.label_standardScaler_obj = None
        # 特征哑编码
        if self.features_onehot_columns is not None:
            self.unique_value(current_iteration_num, train_data, self.features_onehot_columns, self.features_unique_value_list)
        # 特征标准化
        if self.features_standardScaler_columns is not None:
            self.features_standardScaler_obj.partial_fit(train_data[self.features_standardScaler_columns])



    def unique_value(self , current_iteration_num , batch_data , OneHot_columns , unique_value_list):
        """
        功能：分别对每一列统计不重复的值有几个。例如：第一列不重复的值有3个，说明可以分为3个类别，在哑编码后会有3个列
        参数：
            current_iteration_num：使用 pandas 读取大文件时，会分批次读取，current_iteration_num 参数表示当前是第几个批次
            batch_data: 假如当前是第 n 个批次，那么 batch_data 参数表示第 n 个批次时的全部数据
            OneHot_columns: 需要哑编码的列名称
        返回值：无
        """
        if type(OneHot_columns) is not list:
            OneHot_columns = [OneHot_columns]
        for j in range(len(OneHot_columns)):
            single_column_data = batch_data[OneHot_columns[j]]
            unique_value = single_column_data.unique().tolist()
            if (current_iteration_num == 0) and (len(unique_value_list) < len(OneHot_columns)):  # 每一列第一个不重复的值
                unique_value_list.append(unique_value)
            else:
                # 每一列第二个、第三个......不重复的值
                unique_value_list[j].extend(unique_value)  # 在 list 后面追加元素
                # 对每一列中不重复的值去重
                unique_value_list[j] = sorted(set(unique_value_list[j]), key=unique_value_list[j].index)



    def write_csv(self, current_iteration_num, data_dir, train_data, eval_data, test_data, encoding="gbk"):
        train_path = os.path.join(data_dir, "train.csv")
        eval_path = os.path.join(data_dir, "eval.csv")
        test_path = os.path.join(data_dir, "test.csv")
        if current_iteration_num == 0:
            train_data.to_csv(train_path, index=False, encoding=encoding)
            eval_data.to_csv(eval_path, index=False, encoding=encoding)
            test_data.to_csv(test_path, index=False, encoding=encoding)
        else:
            train_data.to_csv(train_path, index=False, encoding=encoding, mode="a", header=False)
            eval_data.to_csv(eval_path, index=False, encoding=encoding, mode="a", header=False)
            test_data.to_csv(test_path, index=False, encoding=encoding, mode="a", header=False)



    def sort_unique_value(self):
        # 哑编码时，对最终的 features_unique_value_list 进行排序。
        if self.features_onehot_columns is not None:
            for i in range(len(self.features_unique_value_list)):
                # 对 unique_value_list[i] 排序
                le = LabelEncoder()
                value_no = le.fit_transform(self.features_unique_value_list[i])  # 字符串转换为数字
                value_no = np.sort(value_no, axis=0)  # 升序
                self.features_unique_value_list[i] = le.inverse_transform(value_no).tolist()  # 反转换
        # 对最终的 label_unique_value_list 进行排序。
        if len(self.label_unique_value_list) > 0:
            le = LabelEncoder()
            value_no = le.fit_transform(self.label_unique_value_list[0])  # 字符串转换为数字
            value_no = np.sort(value_no, axis=0)  # 升序
            self.label_unique_value_list = le.inverse_transform(value_no).tolist()  # 反转换



    def dump_pkl(self, save_dir, pkl_name="数据预处理.pkl"):
        pkl_obj = self.get_pkl_obj()
        pkl_path = os.path.join(save_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_obj, f)



    def print_pkl_obj(self):
        pkl_obj = self.get_pkl_obj()
        print("分类还是回归：" , pkl_obj["classifier_or_regressor"])
        print("列：", pkl_obj["columns"])
        print("record_defaults：", pkl_obj["record_defaults"] , "\n")
        print("标签名：", pkl_obj["label_name"])
        print("标签预处理的方式：", pkl_obj["label_preprocessing_method"])
        print("标签标准化对象：", pkl_obj["label_standardScaler_obj"])
        print("标签不重复的值：", pkl_obj["label_unique_value_list"] , "\n")
        print("特征离散化的列：", pkl_obj["features_discretization_columns"])
        print("特征离散化的bins：", pkl_obj["discretization_bins"] , "\n")
        print("特征哑编码的列：", pkl_obj["features_onehot_columns"])
        print("特征哑编码不重复的值：", pkl_obj["features_unique_value_list"] , "\n")
        print("特征标准化的列：", pkl_obj["features_standardScaler_columns"])
        print("特征标准化对象：", pkl_obj["features_standardScaler_obj"])



    def fit(self, csv_path, chunksize, test_size, eval_size, encoding="gbk"):
        """
        功能：对csv文件进行预处理 fit()，还没有做 transform()，把预处理的规则保存到 pkl 文件
        参数：
            csv_path：csv文件路径
        返回值：无
        """
        # 创建目录，用于保存训练、验证、测试 的数据
        data_dir = self.create_dir(original_csv_path=csv_path)
        with open(csv_path) as file:
            # 大数据时分批读取数据
            dataSet = pd.read_csv(file , chunksize=chunksize , iterator=True)
            i = 0
            for chunk_data in dataSet:
                # 分割训练集、测试集、验证集
                train_data, eval_data, test_data = self.train_test_eval_split(chunk_data, test_size, eval_size)
                # 数据预处理
                self.preprocessing(current_iteration_num=i, train_data=train_data)
                # 把数据写入csv文件
                self.write_csv(i, data_dir, train_data, eval_data, test_data, encoding=encoding)
                i = i + 1
        # 对最终的 unique_value_list 进行排序。
        self.sort_unique_value()
        # 序列化
        self.dump_pkl(data_dir)
        self.print_pkl_obj()

