# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # 分割训练集与测试集
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import dump #保存模型
import threading
import tensorflow as tf




def _int64_feature(value):
    """功能：生成整数型的属性"""
    if not isinstance(value , list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def _bytes_feature(value):
    """功能：生成字符串型的属性"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def convert_to_example(image_buffer , image_label):
    """
    功能：定义图片样例的格式、数据类型
    参数：
        image_buffer：字符串类型的图片数据，相当于机器学习中的 X
        image_label：字符串类型的图片标签，相当于机器学习中的 Y
    返回值：单张图片样例
    """
    example = tf.train.Example(features=tf.train.Features(feature={
        "image_buffer": _bytes_feature(tf.compat.as_bytes(image_buffer)) ,
        "image_label": _bytes_feature(tf.compat.as_bytes(image_label)) ,
    }))
    return example



def create_tfrecord_directory(image_directory):
    """
    功能：分别创建训练集、验证集、测试集的csv文件路径集合
    参数：无
    返回值：原始csv全部文件的路径集合、训练集全部csv文件的路径集合、验证集全部csv文件的路径集合、测试集全部csv文件的路径集合
    """
    # 原始csv文件的全部路径，保存在 original_csv_path_list 集合中
    root_driectory = os.path.dirname(image_directory)
    # 创建训练集、验证集、测试集的文件夹目录
    trainSet_driectory = root_driectory + r"\trainSet"
    validationSet_driectory = root_driectory + r"\validationSet"
    testSet_driectory = root_driectory + r"\testSet"
    driectory_list = [trainSet_driectory , validationSet_driectory , testSet_driectory]
    for i in range(len(driectory_list)):
        is_exists = os.path.exists(driectory_list[i])  # 判断一个目录是否存在
        if is_exists is False:
            os.makedirs(driectory_list[i])  # 创建目录
    return trainSet_driectory , validationSet_driectory , testSet_driectory



def get_tfrecord_file_num(original_file_total_size , each_tfrecord_file_size , test_size , validation_size):
    """
    功能：根据原始数据的大小(占用磁盘空间)、还有每个tfrecord文件的大小，计算出训练集、验证集、测试集分别有多少个 tfrecord 文件
    参数：
        original_file_total_size：原始数据的大小(占用磁盘空间)
        each_tfrecord_file_size：每个tfrecord 文件的大小(占用磁盘空间)
    返回值：训练集、验证集、测试集分别有多少个 tfrecord 文件
    """
    tfrecord_file_total_num = round(original_file_total_size / each_tfrecord_file_size)  # 总共有多少个 tfrecord 文件
    testSet_rate = test_size  # 测试集在全部数据中的比例
    trainSet_rate = 1 - test_size
    validationSet_rate = round(trainSet_rate * validation_size, 2)  # 验证集在全部数据中的比例
    trainSet_rate = round(trainSet_rate - validationSet_rate, 2)  # 训练集在全部数据中的比例
    # 训练集、验证集、测试集分别有多少个 tfrecord 文件
    trainSet_output_file_num = round(trainSet_rate * tfrecord_file_total_num)
    validationSet_output_file_num = round(validationSet_rate * tfrecord_file_total_num)
    testSet_output_file_num = round(testSet_rate * tfrecord_file_total_num)
    # 如果 tfrecord 文件数量等于0 ， 那么 tfrecord 文件数量设置为 1
    if trainSet_output_file_num == 0:
        trainSet_output_file_num = 1
    if validationSet_output_file_num == 0:
        validationSet_output_file_num = 1
    if testSet_output_file_num == 0:
        testSet_output_file_num = 1
    return trainSet_output_file_num, validationSet_output_file_num, testSet_output_file_num



def get_threads_num(trainSet_output_file_num , validationSet_output_file_num , testSet_output_file_num):
    """
    功能：训练集、验证集、测试集分别有多少个线程(根据训练集、验证集、测试集分别有多少个 tfrecord 文件来决定)
    参数：
        trainSet_output_file_num：训练集有多少个 tfrecord 文件
        validationSet_output_file_num：验证集有多少个 tfrecord 文件
        testSet_output_file_num：测试集有多少个 tfrecord 文件
    返回值：训练集、验证集、测试集分别有多少个线程
    """
    trainSet_threads_num = round(0.5 * trainSet_output_file_num)
    validationSet_threads_num = round(0.5 * validationSet_output_file_num)
    testSet_threads_num = round(0.5 * testSet_output_file_num)
    if trainSet_threads_num < 1:
        trainSet_threads_num = 1
    if validationSet_threads_num < 1:
        validationSet_threads_num = 1
    if testSet_threads_num < 1:
        testSet_threads_num = 1
    return trainSet_threads_num, validationSet_threads_num, testSet_threads_num



def read_image_path_and_label(image_directory, test_size, validation_size):
    """
    功能：从图片的根目录，读取 "图片路径" 、"图片标签"。然后分割出训练集、验证集、测试集
    参数：
        image_directory：图片的根目录
        test_size：分割测试集的比例
        validation_size：分割验证集的比例
    返回值：
        训练集(包含：["图片路径" ，"图片标签"])
        验证集(包含：["图片路径" ，"图片标签"])
        测试集(包含：["图片路径" ，"图片标签"])
    """
    label_str_list = []  # 全部图片的标签集合
    image_path_list = []  # 全部图片的路径集合
    unique_value_list = []
    listdir = os.listdir(image_directory)  # 列出文件夹下所有的目录与文件
    for i in range(len(listdir)):
        path = image_directory + "\\" + listdir[i]  # driectory 文件夹下的第 i 个文件或者目录 的路径
        is_dir = os.path.isdir(path)  # 判断是不是目录，如果是目录，返回True。
        if is_dir:
            unique_value_list.append(path.split("\\")[-1])
            try:
                current_directory_all_image_path = path + "\*"
                current_directory_all_image = tf.gfile.Glob(current_directory_all_image_path)  #读取当前目录中的所有图片文件
            except:
                print("图片有问题：" , path)
                continue
            label_str_list.extend([listdir[i]] * len(current_directory_all_image))
            image_path_list.extend(current_directory_all_image)
    dataSet = pd.DataFrame({"图片路径":image_path_list , "标签":label_str_list})
    trainSet, testSet = train_test_split(dataSet, test_size=test_size, shuffle=True, random_state=42)
    trainSet, validationSet = train_test_split(trainSet, test_size=validation_size, shuffle=True, random_state=42)
    # 洗牌后下标是乱的，这时候对下标从新排序，解决了下标乱的问题
    trainSet = trainSet.reset_index(drop=True)
    testSet = testSet.reset_index(drop=True)
    validationSet = validationSet.reset_index(drop=True)
    return trainSet , validationSet , testSet , unique_value_list


def unique_value(unique_value_list):
    # 对 unique_value_list 排序
    le = LabelEncoder()
    value_no = le.fit_transform(unique_value_list)  # 字符串转换为数字
    value_no = np.sort(value_no, axis=0)  # 升序
    unique_value_list = le.inverse_transform(value_no).tolist()  # 反转换
    return unique_value_list


def save_processing_obj(original_image_driectory , unique_value_list , pkl_name="数据预处理.pkl"):
    """
    功能：把 self.preprocessing_obj_dict 保存到模型路径下
    参数：
        pkl_name：pkl文件的名字
    返回值：无
    """
    root_driectory = os.path.dirname(original_image_driectory)
    processing_pkl_Path = root_driectory + "\\" + pkl_name
    preprocessing_obj_dict = {"unique_value" : unique_value_list}
    dump(preprocessing_obj_dict , processing_pkl_Path , compress=3)  # 持久化



def get_threading_range(dataSet, output_file_num, threads_num):
    """
    功能：计算每个线程的范围(每个线程负责的样本数)
        例如：数据集有1000个样本，需要输出10个 tfrecord 文件，使用5个线程，那么每个线程负责2个 tfrecord 文件的写入
        线程1范围是[0, 100, 200]，表示线程1负责的样本是从0到199，第1个tfrecord文件包含样本是从0到100，第2个tfrecord文件从100到200
        线程2范围是[200, 300, 400]，表示线程2负责的样本是从200到400，第1个tfrecord文件包含样本是从0到100，第2个tfrecord文件从100到200
        以此类推..........
    参数：
        resnet_me：数据集，用于计算样本数量
        output_file_num：需要输出多少个 tfrecord 文件
        threads_num：使用多少个线程
    返回值：每个线程的范围
    """
    bins = np.linspace(0, dataSet.shape[0], threads_num + 1).astype(np.int)
    threading_range = {}  # 线程范围
    each_thread_output_file_num = int(output_file_num / threads_num)  # 每个线程输出的文件数量
    for i in range(bins.shape[0] - 1):
        key = "线程" + str(i + 1)
        value = [bins[i], bins[i + 1]]
        for j in range(output_file_num):
            threading_range[key] = current_threading_range = np.linspace(value[0] , value[1] , each_thread_output_file_num + 1)
            threading_range[key] = threading_range[key].astype(int)
    return threading_range



def read_image(single_image_path):
    """
    功能：根据单张图片的路径读取单张图片
    参数：
        single_image_path：单张图片的路径
    返回值：字符串类型的图片数据，仅仅是一张图片的数据
    """
    with tf.gfile.FastGFile(single_image_path , 'rb') as f:
        image_buffer = f.read()
    return image_buffer



def png_to_jpg(png_buffer):
    """
    功能：
        png 与 jpg 图片的编解码。
        注意：本函数只是定义编解码的计算图结构，还没有 sess.run()。返回值是Tensor类型,不能直接转换为图片样例，
        在写入 tfrecord 文件之前需要使用 sess.run(本函数返回值 , feed_dict={本函数的参数 : 实际读取到的字符串类型图片}))
    参数：
        png_buffer：placeholder类型，用于定义编解码的计算图结构，写入 tfrecord 文件之前需要使用 sess.run(返回值 , feed_dict)
    返回值：编解码之后的结果(Tensor类型)
    """
    decode_png = tf.image.decode_png(png_buffer , channels=3)
    encode_jpg = tf.image.encode_jpeg(decode_png , format="rgb" , quality=100)
    return encode_jpg



def single_thread_to_tfrecord(threading_range, thread_name, image_path, image_label, output_dir):
    """
    功能：使用单线程把图片样例写入 tfrecord 文件
    参数：
        threading_range：每一个线程的范围。例如：{"线程1":[0, 100, 200] , "线程2":[200, 300, 400]}
        thread_name：当前线程的名称,也就是参数 threading_range 字典中的 key 值集合的中的其中一个，假如"线程1"就是当前线程名称
        image_path：数据集中全部图片的路径
        image_label：数据集中全部图片的标签
        output_dir：tfRecord 文件的输出目录
    返回值：无
    """
    current_thread_output_file_num = threading_range[thread_name].shape[0] - 1  # 当前线程输出的 TFRecord 文件数量
    for i in range(current_thread_output_file_num):
        output_file_no = (int(thread_name[-1]) - 1) * current_thread_output_file_num + i
        output_filename = "image-" + str(output_file_no) + ".tfrecord"
        output_file_path = output_dir + "\\" +  output_filename  # 需要写入的 TFRecord 文件路径
        tfrecord_writer = tf.python_io.TFRecordWriter(output_file_path)  # 实例化 TFRecordWriter，参数是：需要写入的文件路径
        # batch_index 例如：当前线程的范围是[0,  367,  734, 1101, 1468, 1835], 外层第1次循环： batch_index 的值是从 0到367的矩阵
        # 外层第2次循环： batch_index 的值是从 367到734的矩阵。以此类推...........
        batch_index = np.arange(threading_range[thread_name][i] , threading_range[thread_name][i + 1] , dtype=int)
        for index in batch_index:
            single_image_label = image_label[index]  # 一张图片的标签，字符串格式
            single_image_path = image_path[index]  # image_path 是全部图片的路径，single_image_path 一张图片的路径
            single_image_buffer =  read_image(single_image_path)  #读取一张图片
            example = convert_to_example(single_image_buffer , single_image_label)  # 把图片信息转换为 Example
            tfrecord_writer.write(example.SerializeToString())  # 真正的执行写入操作
        tfrecord_writer.close()



def multi_thread_to_tfrecord(dataSet , tfrecord_output_dir , output_file_num , threads_num):
    """
    功能：使用多线程把图片样例写入 tfrecord 文件。需要调用 single_thread_to_tfrecord() 函数
    参数：
        resnet_me：需要写入的数据集(包含：["图片路径" , "图片标签"])
        tfrecord_output_dir：tfRecord 文件的输出目录
        output_file_num：需要输出多少个 tfRecord 文件
        threads_num：使用的线程数量
    返回值：无
    """
    image_path = dataSet["图片路径"]  # 所有图片的路径
    image_label = dataSet["标签"]  # 所有图片的标签
    threading_range = get_threading_range(dataSet, output_file_num, threads_num)  # 获取线程范围
    # 实例化线程管理器，然后启动线程
    coord = tf.train.Coordinator()  # 实例化线程管理器
    # 依次启动线程。每个线程都调用 single_thread_to_tfrecord() 函数
    thread_list = []
    for thread_name , value in threading_range.items():
        args = (threading_range , thread_name , image_path , image_label , tfrecord_output_dir)  # 单个线程的参数
        current_thread = threading.Thread(target=single_thread_to_tfrecord , args=args)  # 定义单个线程
        current_thread.start()  # 启动线程
        thread_list.append(current_thread)
    coord.request_stop()  # 通知其他线程停止程序
    coord.join(thread_list)  # 等待其他线程停止



def fit(original_image_driectory , test_size , validation_size , original_file_total_size , each_tfrecord_file_size):
    """
    功能：把原始图片写入 tfrecord 文件。需要调用 multi_thread_to_tfrecord() 函数
    参数：
        root_driectory：原始图片的根目录
        test_size：测试集的比例
        validation_size：验证集的比例
        original_file_total_size：原始图片的总共大小(占用的磁盘空间)
        each_tfrecord_file_size：每个 tfrecord 文件的大小(占用磁盘空间)
    返回值：无
    """
    trainSet_output_file_num , validationSet_output_file_num , testSet_output_file_num = get_tfrecord_file_num(
        original_file_total_size , each_tfrecord_file_size , test_size , validation_size
    )
    trainSet_threads_num , validationSet_threads_num , testSet_threads_num = get_threads_num(
        trainSet_output_file_num , validationSet_output_file_num , testSet_output_file_num
    )
    trainSet_driectory , validationSet_driectory , testSet_driectory = create_tfrecord_directory(original_image_driectory)
    trainSet , validationSet , testSet , unique_value_list = read_image_path_and_label(
        original_image_driectory , test_size , validation_size
    )
    unique_value_list = unique_value(unique_value_list)  # 类别的集合，有多少个类别，就有多少个不重复的值。用于哑编码
    save_processing_obj(original_image_driectory , unique_value_list)  # 把 unique_value_list 保存到 pkl 文件
    # 分别对训练集、验证集、测试集写入 tfrecord 文件
    multi_thread_to_tfrecord(trainSet , trainSet_driectory , trainSet_output_file_num , trainSet_threads_num)
    multi_thread_to_tfrecord(validationSet , validationSet_driectory , validationSet_output_file_num , validationSet_threads_num)
    multi_thread_to_tfrecord(testSet , testSet_driectory , testSet_output_file_num , testSet_threads_num)




original_image_driectory = r"F:\data\花\原始图片"
test_size = 0.15
validation_size = 0.15
original_file_total_size = 221
each_tfrecord_file_size = 20

fit(original_image_driectory , test_size , validation_size , original_file_total_size , each_tfrecord_file_size)


