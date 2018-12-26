import os
import tensorflow as tf
from resnet_me.resnet_model import Model
from resnet_me.image_pipeline import input_fn



# resnet_size：有6个可以选值：18、34、50、101、152、200
#   bottleneck：是否使用瓶颈块。如果 resnet_size < 50，那么 bottleneck=False。如果 resnet_size >= 50，那么 bottleneck=True。
#   num_classes：几分类
#   num_filters：64
#   kernel_size：最开始的第一个卷积层的卷积核大小为7
#   conv_stride：最开始的第一个卷积层的卷积核步长为2
#   first_pool_size：最开始的第一个池化层的大小为3
#   first_pool_stride：最开始的第一个池化层的步长为2
#   block_sizes：每个模块组有几个模块。
#     例如 resnet_50，那么 block_sizes=[3 , 4 , 6 , 3]
#     例如 resnet_101，那么 block_sizes=[3 , 4 , 23 , 3]
#   block_strides：中间那层的步长。block_strides=[1, 2, 2, 2]
#   final_size：全连接层的神经元个数。如果 resnet_size < 50，那么 final_size=512。如果 resnet_size >= 50，那么 final_size=2048。
#   resnet_version：有2个可选值：1、2。1表示 resnet_v1，2表示 resnet_v2
#   data_format：有3个可选值："channels_last"、"channels_first"、None



def resnet_v2_model(resnet_size, num_classes):
    if resnet_size < 50:
        bottleneck = False
        final_size = 512
    else:
        bottleneck = True
        final_size = 2048
    num_filters = 64
    kernel_size = 7
    conv_stride = 2
    first_pool_size = 3
    first_pool_stride = 2
    block_sizes = get_block_sizes(resnet_size)
    block_strides = [1, 2, 2, 2]
    resnet_version = 2
    data_format = None
    dtype = tf.float32
    model = Model(
        resnet_size , bottleneck , num_classes , num_filters , kernel_size , conv_stride ,
        first_pool_size , first_pool_stride , block_sizes , block_strides ,
        resnet_version , data_format , dtype
    )
    return model


def get_block_sizes(resnet_size):
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }
  try:
    block_sizes = choices[resnet_size]
    return block_sizes
  except KeyError:
    err = ("resnet_size 参数输入错误")
    raise ValueError(err)



def resnet_model_fn(features, labels, mode, params):
  resnet_size = params["resnet_size"]
  num_classes = params["num_classes"]
  weight_decay = params["weight_decay"]
  batch_size = params["batch_size"]
  momentum = params["momentum"]
  loss_scale = params["loss_scale"]

  tf.summary.image("images" , features , max_outputs=6)
  model = resnet_v2_model(resnet_size, num_classes)
  logits = model(features , mode == tf.estimator.ModeKeys.TRAIN)
  logits = tf.cast(logits , tf.float32)
  predictions = {
      "classes": tf.argmax(logits , axis=1),
      "probabilities": tf.nn.softmax(logits , name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode , predictions=predictions , export_outputs={"predict": tf.estimator.export.PredictOutput(predictions)}
    )
  # 交叉熵
  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits , onehot_labels=labels)
  tf.identity(cross_entropy , name="cross_entropy")
  tf.summary.scalar("cross_entropy" , cross_entropy)
  # 如果没有传递loss_filter_fn，则假设我们需要默认行为，即 batch_normalization 变量从损失值中去除
  def exclude_batch_norm(name):
    return "batch_normalization" not in name
  loss_filter_fn = None
  loss_filter_fn = loss_filter_fn or exclude_batch_norm
  # L2正则化的值
  l2_loss = weight_decay * tf.add_n(
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables() if loss_filter_fn(v.name)]
  )
  tf.summary.scalar("l2_loss" , l2_loss)
  # 最终的损失值
  loss = cross_entropy + l2_loss
  # 如果是训练阶段
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    cifar10_or_ImageNet = "ImageNet"
    learning_rate = learning_rate_with_decay(cifar10_or_ImageNet, global_step, num_images, batch_size, base_lr=0.1, warmup=False)
    tf.identity(learning_rate , name="learning_rate")
    tf.summary.scalar("learning_rate" , learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate , momentum=momentum)
    # train_op
    if loss_scale != 1:
      # 计算梯度时，中间张量值很小，它们下溢到0.0，为避免这种情况，我们将损失乘以 loss_scale，使这些张量值乘以 loss_scale
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)
      # 梯度计算完成后，在梯度传递给优化器之前，将其缩放回正确的比例
      unscaled_grad_vars = [(grad / loss_scale, var) for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars , global_step)
    else:
      minimize_op = optimizer.minimize(loss , global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op , update_ops)
  else:
    train_op = None
  # 计算正确率
  if not tf.contrib.distribute.has_distribution_strategy():
      accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1) , predictions['classes'])
  else:
      accuracy = (tf.no_op(), tf.constant(0))
  metrics = {'accuracy': accuracy}
  tf.identity(accuracy[1] , name="train_accuracy")
  tf.summary.scalar("train_accuracy" , accuracy[1])
  # 返回评估器
  return tf.estimator.EstimatorSpec(
      mode=mode , predictions=predictions , loss=loss , train_op=train_op , eval_metric_ops=metrics
  )



def learning_rate_with_decay(cifar10_or_ImageNet, global_step, num_images, batch_size, base_lr=0.1, warmup=False):
  """
  Args:
    cifar10_or_ImageNet: 有2个可选值："ImageNet"、"cifar10"
  """
  # 定义学习率衰减的规则
  batch_denom, boundary_epochs, decay_rates = None, None, None
  if cifar10_or_ImageNet == "ImageNet":
      batch_denom = 256
      boundary_epochs = [30, 60, 80, 90]
      decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]
  elif cifar10_or_ImageNet == "cifar10":
      batch_denom = 128
      boundary_epochs = [91, 136, 182]
      decay_rates = [1, 0.1, 0.01, 0.001]
  initial_learning_rate = base_lr * batch_size / batch_denom
  batches_per_epoch = num_images["train"] / batch_size
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]
  # 根据衰减规则，计算学习率
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, vals)
  if warmup:
      warmup_steps = int(batches_per_epoch * 5)
      warmup_lr = (
              initial_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(
          warmup_steps, tf.float32))
      return tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: learning_rate)
  return learning_rate



def get_distribution_strategy(num_gpus, all_reduce_alg=None):
  """Return a DistributionStrategy for running the model.
  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.
  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  """
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.MirroredStrategy(
          num_gpus=num_gpus,
          cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
              all_reduce_alg, num_packs=2))
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)




def get_train_hooks(every_n_iter=1):
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }
    # 训练钩子
    train_hooks = tf.train.LoggingTensorHook(tensors=tensors_to_log , every_n_iter=every_n_iter)
    return [train_hooks]



def get_num_gpus(num_gpus):
  """ num_gpus = -1 视为全部使用"""
  if num_gpus != -1:
    return num_gpus
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])



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



def eval_input_fn(data_dir, file_name, boxes, min_object_covered, image_size, batch_size, num_epochs=1):
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



def fit(train_dir , validation_dir , file_name , image_size , resnet_size , num_classes , batch_size , weight_decay , momentum ,
        train_epochs , epochs_per_eval , max_train_steps , model_dir , num_gpus , loss_scale=1):
  tf.logging.set_verbosity(tf.logging.INFO)  # 设置日志
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'  # 使用 Winograd 非融合算法可以提供小的性能提升
  # 使用allow_soft_placement = True，这是多GPU所必需
  session_config = tf.ConfigProto(inter_op_parallelism_threads=0 , intra_op_parallelism_threads=0 , allow_soft_placement=True)
  distribution_strategy = get_distribution_strategy(get_num_gpus(num_gpus) , all_reduce_alg=None)  # 分配策略
  # run_config = tf.deep_learning.RunConfig(train_distribute=distribution_strategy, session_config=session_config)
  run_config = tf.estimator.RunConfig(session_config=session_config)
  # 实例化 Estimator 评估器
  classifier = tf.estimator.Estimator(
      model_fn=resnet_model_fn , model_dir=model_dir , config=run_config ,
      params={
          "resnet_size": resnet_size ,
          "num_classes": num_classes ,
          "weight_decay": weight_decay ,
          "batch_size": batch_size ,
          "momentum": momentum ,
          "loss_scale": loss_scale ,
      }
  )
  # 训练
  for _ in range(train_epochs // epochs_per_eval):
    train_hooks = get_train_hooks(every_n_iter=1)  # 训练钩子
    # 使用训练集训练
    classifier.train(
        input_fn = lambda: train_input_fn(
            data_dir=train_dir, file_name=file_name, boxes=None, min_object_covered=1.0,
            image_size=image_size, batch_size=batch_size, num_epochs=epochs_per_eval
        ) ,
        hooks = train_hooks ,
        # steps = 1
        # max_steps = max_train_steps
    )
    # 使用验证集评估
    eval_results = classifier.evaluate(
        input_fn =lambda: eval_input_fn(
            data_dir=validation_dir, file_name=file_name, boxes=None, min_object_covered=1.0,
            image_size=image_size, batch_size=batch_size, num_epochs=1
        ) ,
        # steps = 1
    )
    tf.logging.info("验证结果：%s" , eval_results)
    print("验证结果：" , eval_results)






num_images = {
    "train": 2652 ,
    "validation": 468 ,
}


train_dir = r"F:\data\花\trainSet"
validation_dir = r"F:\data\花\validationSet"
file_name = "image-*.tfrecord"
image_size = [28 , 28 , 3]
num_classes = 5

resnet_size = 18
batch_size = 32
weight_decay = 1e-4
momentum = 0.9
train_epochs = 90
epochs_per_eval = 1
max_train_steps = None
model_dir = r"C:\Users\lenovo\Desktop\resnet_model"
num_gpus = 1 if tf.test.is_gpu_available() else 0

fit(
    train_dir, validation_dir, file_name, image_size, resnet_size, num_classes, batch_size, weight_decay, momentum,
    train_epochs, epochs_per_eval, max_train_steps, model_dir, num_gpus, loss_scale=1
)

