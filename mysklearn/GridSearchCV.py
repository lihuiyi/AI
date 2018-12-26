# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 分割训练集与测试集




#功能：从csv文件中读取数据，保存到dataSet中
#参数：csv路径
#返回值：读取到的数据集（DataFrame格式）
def loadData(csvPath):
    csvFile = open(csvPath)
    dataSet = pd.read_csv(csvFile)
    csvFile.close()
    return dataSet




def main():
    #从文件中读取数据，返回DataFrame格式的数据集
    ProjectPath = os.path.abspath('.') #动态获取项目路径
    filePath = ProjectPath + r"\data\鸢尾花.csv"
    dataSet = loadData(filePath)
    
    
    #分割训练集和测试集
    x = dataSet.loc[: , dataSet.columns != "标签"]
    y = dataSet.loc[: , dataSet.columns == "标签"]
    replace_dict = {"Iris-virginica":0 , "Iris-versicolor":1 , "Iris-setosa":2}
    y = y.replace(replace_dict)
    x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.3 , shuffle=False, random_state=0)
    
    
    
    #自动化调参
#     from sklearn.pipeline import Pipeline
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.model_selection import GridSearchCV
#     import psutil
#     lr = LogisticRegression(penalty = "l2")
#     module = ('LogisticRegression', lr)
#     pipeline = Pipeline(steps=[module]) #流水线处理
#     #新建网格搜索对象
#     #第一参数为待训练的模型
#     #param_grid为待调参数组成的网格，字典格式，键为参数名称（格式“对象名称__子对象名称__参数名称”），值为可取的参数值列表
#     param_test = [
#                   {"LogisticRegression__solver":["newton-cg", "lbfgs" , "liblinear"]} , 
#                   {"LogisticRegression__C":[0.01 , 0.1 , 10 , 100]} 
#                   ]
#     grid_search = GridSearchCV(estimator = pipeline , param_grid = param_test , cv=5 , verbose=1)
#     #训练以及调参
#     grid_search.fit(x_train , y_train.values.ravel())
#     print("\n\n" , grid_search.cv_results_ , "\n\n") #给出不同参数情况下的评价结果
#     print(grid_search.best_params_ , "\n\n") #描述了已取得最佳结果的参数的组合
#     print(grid_search.best_score_ , "\n\n") #成员提供优化过程期间观察到的最好的评分
#     #获取内存占用率
#     a = psutil.virtual_memory().percent
#     print ('获取内存占用率： '+ str(a) + "%")
     
     
     
    #把训练好的模型保存起来
#     from sklearn.externals.joblib import dump
#     from sklearn.externals.joblib import load
#     from sklearn.linear_model import LogisticRegression
#     #第一个参数训练好的模型，第二个参数文件路径 ，第三个参数是压缩级别，0为不压缩，3为合适的压缩级别
#     fileName = "鸢尾花模型.dmp"
#     filePath = r"C:\Users\Think\Desktop\\" + fileName
#     lr = LogisticRegression()
#     lr.fit(x_train , y_train.values.ravel())
#     dump(lr , filePath , compress=0)
#     print("完成")

    
    #加载模型到内存中，然后进行预测
#     from sklearn.externals.joblib import dump
#     from sklearn.externals.joblib import load
#     fileName = "鸢尾花模型.dmp"
#     filePath = r"C:\Users\Think\Desktop\\" + fileName
#     lr = load(filePath)
#     y_predict = lr.predict(x_test)
#     print(y_predict == y_test.values.ravel())
    
    

main()
    
    
    
    