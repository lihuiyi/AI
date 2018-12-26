# -*- coding: utf-8 -*-

import tensorflow as tf
from random import choice



R_mean = 123.68
G_mean = 116.78
B_mean = 103.94
channel_means = [R_mean, G_mean, B_mean]
# 保留图像最小边的下限。 例如，如果图像为500 x 1000，则会将其调整为 [resize_min , 2 * resize_min , channel]
resize_min = 256
label_unique_value_list = "label_unique_value_list"



# def read_image(tfrecord_file_path , num_epochs , is_training):
#     file_list = tf.train.match_filenames_once(tfrecord_file_path)
#     if is_training:
#         filename_queue = tf.train.string_input_producer(file_list , shuffle=True , num_epochs=num_epochs)
#     else:
#         filename_queue = tf.train.string_input_producer(file_list, shuffle=False , num_epochs=1)
#     reader = tf.TFRecordReader()
#     _, serialized_example = reader.read(filename_queue)
#     features = tf.parse_single_example(serialized_example , features={
#         "image_buffer": tf.FixedLenFeature([], tf.string) ,
#         "image_label": tf.FixedLenFeature([], tf.string) ,
#     })
#     image_buffer = features["image_buffer"]
#     image_label = features["image_label"]
#     return image_buffer , image_label



def decode_crop_and_flip(image_buffer, bbox, min_object_covered, num_channels):
  """
  功能：将给定图像裁剪为图像的随机部分，并随机翻转我们使用融合的decode_and_crop op，其性能优于串联单独使用的两个op，
    图像包含人注释的边框，创建一个随机的新边框，大小和与人类注释的重叠边框。 如果没有边框，那么边框是整个图片
  参数：
    image_buffer：没有解码的图片
    bbox：标注框的坐标，[ymin, xmin, ymax, xmax]
    min_object_covered：float类型，表示截取后的图像至少包含某个标注框百分之 min_object_covered 的信息
    num_channels：图片通道数
  返回值：裁剪后的图片
  """
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      tf.image.extract_jpeg_shape(image_buffer),
      bounding_boxes=bbox,
      min_object_covered=min_object_covered,
      aspect_ratio_range=[0.75, 1.33],
      area_range=[0.05, 1.0],
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box
  # 以 crop op 所需的格式重新组装边界框。
  offset_y, offset_x, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
  # 在这里使用融合解码和裁剪操作，这比串联中的每个更快。
  cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)
  # 图像随机翻转
  cropped = flip_image(cropped)
  return cropped



def get_boxes(coordinate=None):
    """
    功能：获取标注框的坐标
    参数：
        coordinate：表示注框的坐标
            如果 coordinate=None，那么返回值=[]。如果只有一个标注框，是 list类型，如果有多高标注框，是 list of list 类型
    返回值：注框的坐标(Tensor类型)
    """
    if coordinate is None:
        xmin = tf.expand_dims([], 0)
        ymin = tf.expand_dims([], 0)
        xmax = tf.expand_dims([], 0)
        ymax = tf.expand_dims([], 0)
        bbox = tf.concat([ymin, xmin, ymax, xmax], 0)
        bbox = tf.expand_dims(bbox, 0)
        bbox = tf.transpose(bbox, [0, 2, 1])
    else:
        if len(tf.constant(coordinate).shape) == 1:
            bbox = tf.constant([[coordinate]])
        else:
            bbox = tf.constant([coordinate])
    return bbox



def flip_image(image):
    flip_method_list = ["left_right", "up_down"]
    choice_one_item = choice(flip_method_list)  # 从 flip_method_list 中随机选择其中一个元素
    if choice_one_item == "left_right":
        cropped = tf.image.random_flip_left_right(image)  # 随机左右翻转
    elif choice_one_item == "up_down":
        cropped = tf.image.random_flip_up_down(image)  # 随机上下翻转
    return image



def central_crop(image, crop_height, crop_width):
  """
  功能：中央裁剪
  参数：
    image：解码后的图片数据
    crop_height：裁剪后的图像高
    crop_width：裁剪后的图像宽
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]
  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])



def mean_image_subtraction(image, means, num_channels):
  """
  功能：从每个通道中减去给定的平均值。例如：R通道减给定的平均值，G通道减给定的平均值，B通道减给定的平均值
      平均值 = [123.68, 116.779, 103.939]
  参数：
    image：解码后的图片数据
    means：平均值
    num_channels：通道数
  返回值：居中的图片
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')
  # 把1维张量，转换为3维张量
  means = tf.expand_dims(tf.expand_dims(means, 0), 0)
  return image - means



def smallest_size_at_least(height, width, resize_min):
  """
  功能：计算最小边等于“smallest_side”的新形状，保留原始宽高比。
  参数：
    height：图片的高，Tensor类型
    width：图片的宽，Tensor类型
    resize_min：int类型，表示调整后最短的那一边的大小。
  """
  resize_min = tf.cast(resize_min, tf.float32)
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)
  return new_height , new_width



def aspect_preserving_resize(image, resize_min):
  """
  功能：调整图像大小，保留原始宽高比。
  参数：
    image：解码后的图片数据
    resize_min: int类型，表示调整后最短的那一边的大小。
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]
  new_height, new_width = smallest_size_at_least(height, width, resize_min)
  image = resize_image(image, new_height, new_width)
  return image



def resize_image(image, height, width):
    """
    功能：调整图像大小
    参数：
        image：解码后的图片数据
        height: 图片的高
        width: 图片的宽
    返回值：调整后的图片
    """
    image = tf.image.resize_images(image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
    return image



def adjust_image_color(image):
    adjust_method_list = [None , "brightness" , "contrast" , "hue" , "saturation"]  #调整颜色方法的集合
    adjust_method = choice(adjust_method_list)  #从 adjust_method_list 中随机选择其中一个元素
    if adjust_method is None:
        return image
    elif adjust_method == "brightness":
        image = tf.image.random_brightness(image , max_delta=0.5)  # 调整图像亮度
    elif adjust_method == "contrast":
        image = tf.image.random_contrast(image , lower=0.5 , upper=1.0)  # 调整图像对比度
    elif adjust_method == "hue":
        image = tf.image.random_hue(image , max_delta=0.1)  # 调整图像色相
    elif adjust_method == "saturation":
        image = tf.image.random_saturation(image , lower=0.2 , upper=1.8)  # 调整图像饱和度
    return image



def preprocess_image(image_buffer, boxes, min_object_covered, output_image_size, is_training=False):
  """
  功能：预处理图片
  参数：
    image_buffer：没有解码的图片
    bbox：标注框的坐标，[ymin, xmin, ymax, xmax]
    min_object_covered：float类型，表示截取后的图像至少包含某个标注框百分之 min_object_covered 的信息
    output_image_size：预处理后的图片大小，格式：[高 , 宽 , 颜色通道]
    is_training：是否是训练阶段
  返回值：预处理后的图片
  """
  output_height, output_width, num_channels = output_image_size[0] , output_image_size[1] , output_image_size[2]
  if is_training:
    bbox = get_boxes(boxes)
    # bbox = boxes
    image = decode_crop_and_flip(image_buffer, bbox, min_object_covered, num_channels)
    image = resize_image(image, output_height, output_width)
    image = adjust_image_color(image)  # 调整图像颜色
  else:
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
    image = aspect_preserving_resize(image, resize_min)
    image = central_crop(image, output_height, output_width)
  image.set_shape([output_height, output_width, num_channels])
  image = mean_image_subtraction(image, channel_means, num_channels)
  return image



def labelencoder(data , unique_value):
    data = tf.py_func(_labelencoder_func, [data , unique_value], tf.int32)
    return data


def _labelencoder_func(data , unique_value):
    data = unique_value.tolist().index(data)
    return data



def label_onehot(label, pkl_obj):
    # unique_value = pkl_obj[label_unique_value_list][0]
    unique_value = ["daisy" , "dandelion" , "roses" , "sunflowers" , "tulips"]
    label = labelencoder(label , unique_value)
    label = tf.one_hot(label, len(unique_value), 1, 0)
    return label

