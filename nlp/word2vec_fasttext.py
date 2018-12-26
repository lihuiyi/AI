# -*- coding: utf-8 -*-

import os
import pandas as pd
import jieba
import jieba.analyse
import codecs



def read_stopwords(stop_words_path):
    with open(stop_words_path  , encoding="utf-8", errors="ignore") as file:
        columns = "停用词"
        dataSet = pd.read_csv(file , index_col=False , sep="\t" , quoting=3 , names=[columns])
    stop_words = dataSet["停用词"].tolist()
    stop_words = set(stop_words)
    return stop_words



def cut_word(source_file_path, target_file_path, stop_words_path, source_label_splitter="\t"):
    """
    功能：分词
    参数
        source_file_path：分词前的文件路径
        target_file_path：分词后的文件路径
    """
    stop_words = read_stopwords(stop_words_path)
    # 打开文件
    source_file = codecs.open(source_file_path, "r", encoding="utf-8")
    target_file = codecs.open(target_file_path, "w", encoding="utf-8")
    line = source_file.readline()  # 读取一行
    line_num = 0  # 处理的行数
    category_list = []
    while line:
        # 去掉空格、换行符
        line = line.replace("\r", "").replace("\n", "").strip()
        # 如果 line 为空，就跳过这次循环，进入下一次循环
        if line == "" or line is None:
            line = source_file.readline()  # 读取下一行
            continue
        line_list = line.split(source_label_splitter)
        if len(line_list) != 2:
            line = source_file.readline()  # 读取下一行
            continue
        label = line_list[0]
        feature = line_list[-1]
        # 分词
        line_seg = jieba.lcut(feature)
        # 去停用词
        line_seg = filter(lambda word: word not in stop_words, line_seg)
        line_seg = " ".join(line_seg)
        # fasttext 特有的步骤，构造 "__label__" + label + "\t" + line_seg + "\r\n"
        line_seg = "__label__" + label + "\t" + line_seg + "\r\n"
        # 把每一行写入 target_file 文件
        target_file.writelines(line_seg)
        # 每次循环都要读取下一行
        line = source_file.readline()
        if label not in category_list:
            category_list.append(label)
        # 打印进度
        if (line_num % 10000 == 0) and (line_num >= 10000):
            print("第" + str(line_num) + "行分词完成")
        line_num = line_num + 1
    # 关闭文件
    source_file.close()
    target_file.close()
    print("分词完成")







data_dir = r"C:\Users\lenovo\Desktop\data\新闻分类"
source_file_path = os.path.join(data_dir, "新闻分类训练集.txt")
target_file_path = os.path.join(data_dir, "分词后的新闻分类训练集.txt")
stop_words_path = os.path.join(os.path.dirname(data_dir), "stop_words.txt")
model_dir = r"F:\model\新闻分类"

# 分词
cut_word(source_file_path , target_file_path, stop_words_path)



