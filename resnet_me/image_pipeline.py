# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from resnet_me.image_preprocessing import preprocess_image
from resnet_me.image_preprocessing import label_onehot



def get_filenames(data_dir , file_name):
    file_path = os.path.join(data_dir, file_name)
    filenames = tf.gfile.Glob(file_path)
    return filenames



def parse_example_proto(serialized_example):
    features = tf.parse_single_example(serialized_example, features={
        "image_buffer": tf.FixedLenFeature([], tf.string),
        "image_label": tf.FixedLenFeature([], tf.string),
    })
    image_buffer = features["image_buffer"]
    image_label = features["image_label"]
    return image_buffer, image_label



def map_fn(serialized_example , boxes , min_object_covered , image_size , is_training):
    image_buffer, label = parse_example_proto(serialized_example)
    image = preprocess_image(image_buffer , boxes , min_object_covered , image_size , is_training)
    # 标签哑编码
    label = label_onehot(label, pkl_obj=None)
    return image , label



def input_fn(is_training , data_dir , file_name , boxes, min_object_covered, image_size , batch_size ,
             num_epochs , shuffle_buffer=10000):
    filenames = get_filenames(data_dir, file_name)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if is_training:
        # 对输入文件洗牌
        dataset = dataset.shuffle(buffer_size=len(filenames))
    # 并行地读取并解析多个数据文件
    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10)
    )
    # 读取数据和训练是并行
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.repeat(num_epochs)
    # map_and_batch
    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(
            lambda value: map_fn(value, boxes , min_object_covered , image_size , is_training),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=False ,
        )
    )
    # 读取数据和训练是并行
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset



def train_input_fn(data_dir , file_name , boxes , min_object_covered , image_size , batch_size , num_epochs):
    return input_fn(
        is_training=True,
        data_dir=data_dir,
        file_name=file_name,
        boxes=boxes,
        min_object_covered=min_object_covered,
        image_size=image_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle_buffer=10000
    )



def eval_input_fn(data_dir , file_name , boxes , min_object_covered , image_size , batch_size , num_epochs=1):
    return input_fn(
        is_training=False,
        data_dir=data_dir,
        file_name=file_name,
        boxes=boxes,
        min_object_covered=min_object_covered,
        image_size=image_size,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle_buffer=10000
    )




# train_dir = r"F:\data\花\trainSet"
# file_name =r"image-*.tfrecord"
# image_size = [224,224,3]
# batch_size = 64
# num_epochs = 1
#
# boxes = None
# min_object_covered = 1.0
#
# num_images = {
#     "train": 2651 ,
#     "validation": 1000 ,
# }
#
# dataset = input_fn(
#             is_training=True, data_dir=train_dir, file_name=file_name, boxes=boxes, min_object_covered=min_object_covered,
#             image_size=image_size, batch_size=batch_size , num_epochs=num_epochs
#         )
# iterator = dataset.make_one_shot_iterator()
# dataset = iterator.get_next()
#
# from datetime import datetime
# s = datetime.now()
#
# with tf.Session() as sess:
#     i = 0
#     while True:
#         try:
#             data = sess.run(dataset)
#             # image_data = data[0]
#             # label_data = data[1]
#             # print(image_data , label_data)
#             print(i)
#             i = i + 1
#
#             # path = r"C:\Users\lenovo\Desktop\新建文件夹\\"
#             # for i in range(batch_size):
#             #     encode_jpeg = tf.image.encode_jpeg(image_data[i] , format="rgb" , quality=100)
#             #     with tf.gfile.GFile(path + str(i) + ".jpg" , "wb") as f:
#             #         f.write(encode_jpeg.eval())
#             # break
#
#         except tf.errors.OutOfRangeError:
#             break
# e = datetime.now()
# print(e - s)