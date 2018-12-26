# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import codecs
import tensorflow as tf
import collections



batch_size = 64
learn_rate = 0.01
model_dir = r"F:\model\唐诗"
data_dir = r"F:\data\唐诗"
file_path = os.path.join(data_dir, "唐诗.txt")
epoch = 50

start_token = "G"
end_token = "E"



def process_pomes(file_path, source_label_splitter=":"):
    """
    功能：数据预处理
    参数
        file_path：文件路径
        source_label_splitter：标题与正文之间的分割符号
    返回值：pomes_vector, word2vec_map, words
        pomes_vector：pomes_vector 是list，每一个元素是一段正文的所有词组成的大型词向量
        word2vec_map：是一个字典，每一个词的词向量，key是词，value是词向量
        words：全部词汇
    """
    poems = []
    source_file = codecs.open(file_path, "r", encoding="utf-8")
    # target_file = codecs.open(target_file_path, "w", encoding="utf-8")
    line = source_file.readline()  # 读取一行
    while line:
        # 去掉空格、换行符
        line = line.replace("\r", "").replace("\n", "").strip()
        # 如果 line 为空，就跳过这次循环，进入下一次循环
        if line == "" or line is None:
            line = source_file.readline()  # 读取下一行
            continue
        # 拆分标题、和正文
        line_list = line.split(source_label_splitter)
        if len(line_list) != 2:
            line = source_file.readline()  # 读取下一行
            continue
        title = line_list[0]
        content = line_list[-1]
        # 如果正文的词数量太少或者太多，就跳出这次循环
        if len(content) < 5 or len(content) > 80:
            line = source_file.readline()  # 读取下一行
            continue
        # 构造句子的内容，以 start_token 为开始，以 content 为正文，以 end_token 为结尾
        content = start_token + content + end_token
        poems.append(content)
        line = source_file.readline()  # 读取一行
    poems = sorted(poems, key=lambda le : len(line))  # 排序
    # 计算所有词
    all_words = []
    for poem in poems:
        all_words += [word for word in  poem]
    # 统计词频
    counter = collections.Counter(all_words)
    # 过滤掉低频词、生僻词
    counter_paris = sorted(counter.items(), key=lambda x:x[-1])
    words, _ = zip(*counter_paris)
    words = words[: len(words)]

    # 获取每一个词的词向量，word2vec_map 是一个字典，key是词，value是词向量
    word2vec_map = dict(zip(words, range(len(words))))
    # pomes_vector 是list，每一个元素是一段正文的所有词组成的大型词向量
    to_num = lambda word: word2vec_map.get(word, len(words))
    pomes_vector = [list(map(to_num, poem)) for poem in poems]
    return pomes_vector, word2vec_map, words



def get_batch_data(batch_size, pomes_vector, word2vec_map):
    n_chunk = len(pomes_vector) / batch_size
    n_chunk = int(n_chunk)
    x_batchs = []
    y_batchs = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        batchs = pomes_vector[start_index : end_index]
        # 当前batch数据中，所有句子中最大长度是多少
        lenght = max(map(len, batchs))
        x_data = np.full((batch_size, lenght), 0, np.float32)  # 填充维度，使用0填充
        for row in range(batch_size):
            x_data[row, :len(batchs[row])] = batchs[row]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        x_batchs.append(x_data)
        y_batchs.append(y_data)
    return x_batchs, y_batchs



def run_training(file_path):
    pomes_vector, word2vec_map, words = process_pomes(file_path, source_label_splitter=":")
    x_batchs, y_batchs = get_batch_data(batch_size, pomes_vector, word2vec_map)
    intput_data = tf.placeholder(tf.int32, [batch_size, None])
    output_data = tf.placeholder(tf.int32, [batch_size, None])





def main(is_training, file_path):
    if is_training:
        print("开始训练")
        run_training(file_path)
    else:
        begin_word = input("输入开始词：")








