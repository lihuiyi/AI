# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


import os
import re
import jieba
import jieba.analyse
import codecs
import logging
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import PathLineSentences
import multiprocessing
import pandas as pd



def read_stopwords(stop_words_path):
    with open(stop_words_path  , encoding="utf-8", errors="ignore") as file:
        columns = "停用词"
        dataSet = pd.read_csv(file , index_col=False , sep="\t" , quoting=3 , names=[columns])
    stop_words = dataSet["停用词"].tolist()
    stop_words = set(stop_words)
    return stop_words



def cut_word(source_file_path , target_file_path):
    """
    功能：分词
    参数
        source_file_path：分词前的文件路径
        target_file_path：分词后的文件路径
    """
    # 打开文件
    source_file = codecs.open(source_file_path, "r", encoding="utf-8")
    target_file = codecs.open(target_file_path, "w", encoding="utf-8")
    line = source_file.readline()  # 读取一行
    line_num = 0  # 处理的行数
    while line:
        # 去掉空格、换行符
        line = line.replace("\r", "").replace("\n", "").strip()
        # 去掉标点符号、特殊字符、数字、英文等字符，只保留中文
        pattern = re.compile(r'[\u4e00-\u9fa5]+')
        filter_data = re.findall(pattern, line)
        line = "".join(filter_data)
        # 如果 line 为空，就跳过这次循环，进入下一次循环
        if line == "" or line is None:
            line = source_file.readline()  # 读取下一行
            continue
        # 分词
        line_seg = " ".join(jieba.cut(line)) + "\r\n"
        target_file.writelines(line_seg)  # 把每一行写入 target_file 文件
        # 每次循环都要读取下一行
        line = source_file.readline()
        # 打印进度
        if (line_num % 10000 == 0) and (line_num >= 10000):
            print("第" + str(line_num) + "行分词完成")
        line_num = line_num + 1
    # 关闭文件
    source_file.close()
    target_file.close()
    print("分词完成")



def train(file_path, model_dir, size, window, min_count, iter=5):
    """
    功能：训练 word2vec 词向量模型
    参数：
        file_path：分词后的文件
        model_path：模型保存路径
        vector_path：词向量保存路径
        size：词向量的维度
        window：上下文最大距离
        min_count：计算最小词频，可以去掉一些很生僻的低频词
        iter：最大次数，默认是5
    返回值：无
    """
    # 创建 model_path
    is_exists = os.path.exists(model_dir)  # 判断一个目录是否存在
    if is_exists is False:
        os.makedirs(model_dir)  # 创建目录
    model_path = os.path.join(model_dir, "model.bin")
    vector_path = os.path.join(model_dir, "vector.bin")
    # 日志
    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    # 训练
    model = word2vec.Word2Vec(
        LineSentence(file_path),
        size=size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        iter=iter
    )
    # 保存模型
    model.save(model_path)
    model.wv.save_word2vec_format(vector_path, binary=False)
    print("训练完成")



def predict(model_dir , word):
    """
    功能：预测
    参数：
        model_path：模型路径
        word：需要预测的词(list类型)
    返回值：无
    """
    model_path = os.path.join(model_dir, "model.bin")
    model = word2vec.Word2Vec.load(model_path)
    for i in range(len(word)):
        try:
            res = model.most_similar(word[i])
            print(word[i], res)
        except KeyError:
            print(word[i] + " 不在模型中")






data_dir = r"C:\Users\lenovo\Desktop\data\笑傲江湖"
source_file_path = os.path.join(data_dir, "笑傲江湖.txt")
target_file_path = os.path.join(data_dir, "分词后的笑傲江湖.txt")
stop_words_path = os.path.join(os.path.dirname(data_dir), "stop_words.txt")
model_dir = r"F:\model\笑傲江湖"

# # 分词
# cut_word(source_file_path , target_file_path)

# # 训练
# train(target_file_path, model_dir, size=100, window=5, min_count=5)

# # 预测
# word = ["令狐冲", "东方不败", "岳不群", "任我行"]
# predict(model_dir , word)

# # 计算2个词的相似度
# model_path = os.path.join(model_dir, "model.bin")
# model = word2vec.Word2Vec.load(model_path)
# sim = model.similarity("令狐冲", "岳不群")
# print(sim)

# # 计算2个集合的相似度
# model_path = os.path.join(model_dir, "model.bin")
# model = word2vec.Word2Vec.load(model_path)
# list1 = ["令狐冲", "东方不败"]
# list2 = ["岳不群", "任我行"]
# sim_n = model.n_similarity(list1, list2)
# print(sim_n)

# # 从集合中选出不相似的词
# model_path = os.path.join(model_dir, "model.bin")
# model = word2vec.Word2Vec.load(model_path)
# list3 = ["令狐冲", "东方不败", "岳不群", "任我行"]
# sim = model.doesnt_match(list3)
# print(sim)

# # 查看词向量
# model_path = os.path.join(model_dir, "model.bin")
# model = word2vec.Word2Vec.load(model_path)
# vev = model["令狐冲"]
# print(vev, vev.shape)

