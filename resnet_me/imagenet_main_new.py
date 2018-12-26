import tensorflow as tf
from resnet_me import resnet_model
from resnet_me.image_pipeline import train_input_fn, eval_input_fn



###############################################################################
# 定义 Imagenet 模型
###############################################################################
class ImagenetModel(resnet_model.Model):
  """功能：定义 Imagenet 模型"""

  def __init__(self, resnet_size, data_format=None, num_classes=None,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
    """
    功能：设置 Imagenet 默认超参数。
    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
    else:
      bottleneck = True

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype
    )



def _get_block_sizes(resnet_size):
  """
  功能：选择 ResNet 模型中每个 block_layer 的数量，也就是 block_sizes
  Args:
    resnet_size: The number of convolutional layers needed in the model.
  Returns:
    A list of block sizes to use in building the model.
  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)



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



def model_fn(features, labels, mode, params):
  num_images = params["num_images"]
  cifar10_or_ImageNet = params["cifar10_or_ImageNet"]
  fine_tune = params["fine_tune"]
  resnet_version = params["resnet_version"]
  resnet_size = params["resnet_size"]
  # data_format = params["data_format"],
  num_classes = params["num_classes"]
  weight_decay = params["weight_decay"]
  batch_size = params["batch_size"]
  momentum = params["momentum"]
  loss_scale = params["loss_scale"]
  dtype = params["dtype"]

  # 是否微调
  if fine_tune:
    warmup = False
    base_lr = .1
  else:
    warmup = True
    base_lr = .128

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)
  # Checks that features/images have same data type being used for calculations.
  assert features.dtype == dtype

  model = ImagenetModel(resnet_size, data_format=None, num_classes=num_classes, resnet_version=resnet_version, dtype=dtype)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  # cross_entropy = tf.losses.sparse_softmax_cross_entropy(
  #     logits=logits, labels=labels)

  cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = None
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_with_decay(
        cifar10_or_ImageNet, global_step, num_images, batch_size, base_lr, warmup
    )

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=momentum
    )

    def _dense_grad_filter(gvs):
      """Only apply gradient updates to the final layer.
      This function is used for fine tuning.
      Args:
        gvs: list of tuples with gradients and variable info
      Returns:
        filtered gradients so that only the dense layer remains
      """
      return [(g, v) for g, v in gvs if 'dense' in v.name]

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      if fine_tune:
        scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss)
      if fine_tune:
        grad_vars = _dense_grad_filter(grad_vars)
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
  else:
    train_op = None

  # accuracy = tf.metrics.accuracy(labels, predictions['classes'])
  # accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
  #                                                 targets=labels,
  #                                                 k=5,
  #                                                 name='top_5_op'))

  accuracy = tf.metrics.accuracy(tf.argmax(labels, axis=1), predictions['classes'])
  accuracy_top_5 = tf.metrics.mean(tf.nn.in_top_k(predictions=logits,
                                                  targets=tf.argmax(labels, axis=1),
                                                  k=5,
                                                  name='top_5_op'))
  #

  metrics = {'accuracy': accuracy,
             'accuracy_top_5': accuracy_top_5}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
  tf.summary.scalar('train_accuracy', accuracy[1])
  tf.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)



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
  """num_gpus = -1 视为全部使用"""
  if num_gpus != -1:
    return num_gpus
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top
  local_device_protos = device_lib.list_local_devices()
  return sum([1 for d in local_device_protos if d.device_type == "GPU"])



def fit(train_dir, eval_dir, file_name, num_images, num_classes, image_size, model_dir, cifar10_or_ImageNet, fine_tune,
        resnet_size, batch_size, weight_decay, train_epochs, epochs_per_eval, loss_scale=1):
  """Shared main loop for ResNet Models.
  Args:
  """
  tf.logging.set_verbosity(tf.logging.INFO)  # 设置日志
  # 创建会话配置。 allow_soft_placement = True，是必需的。multi-GPU，对其他模式无害。
  session_config = tf.ConfigProto(inter_op_parallelism_threads=0, intra_op_parallelism_threads=0, allow_soft_placement=True)
  run_config = tf.estimator.RunConfig(session_config=session_config)
  # 实例化 Estimator 评估器
  classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=model_dir, config=run_config,
      params={
          "num_images": num_images,
          "num_classes": num_classes,
          "cifar10_or_ImageNet": cifar10_or_ImageNet,
          "fine_tune": fine_tune,
          "resnet_version": 2,
          "resnet_size": resnet_size,
          # "data_format": None,
          "weight_decay": weight_decay,
          "batch_size": batch_size,
          "momentum": 0.9,
          "loss_scale": loss_scale,
          "dtype": tf.float32
      }
  )
  # 训练
  for _ in range(train_epochs // epochs_per_eval):
    train_hooks = get_train_hooks(every_n_iter=1)
    # 使用训练集训练
    classifier.train(
        input_fn=lambda: train_input_fn(
            data_dir=train_dir, file_name=file_name, boxes=None, min_object_covered=1.0,
            image_size=image_size, batch_size=batch_size, num_epochs=epochs_per_eval
        ),
        hooks=train_hooks,
        # steps = 1
        # max_steps = max_train_steps
    )
    # 使用验证集评估
    eval_results = classifier.evaluate(
        input_fn=lambda: eval_input_fn(
            data_dir=eval_dir, file_name=file_name, boxes=None, min_object_covered=1.0,
            image_size=image_size, batch_size=batch_size, num_epochs=1
        ),
        # steps = 1
    )
    tf.logging.info("验证结果：%s", eval_results)
    print("验证结果：", eval_results)




num_images = {
    "train": 2652 ,
    "validation": 468 ,
}

train_dir = r"F:\data\花\trainSet"
eval_dir = r"F:\data\花\validationSet"
file_name = "image-*.tfrecord"
num_classes = 5
image_size = [28 , 28 , 3]
model_dir = r"C:\Users\lenovo\Desktop\resnet_model"
cifar10_or_ImageNet = "ImageNet"
fine_tune = False

resnet_size = 18
batch_size = 32
weight_decay = 1e-4
train_epochs = 90
epochs_per_eval = 1
# num_gpus = 1 if tf.test.is_gpu_available() else 0


fit(
    train_dir, eval_dir, file_name, num_images, num_classes, image_size, model_dir, cifar10_or_ImageNet, fine_tune,
    resnet_size, batch_size, weight_decay, train_epochs, epochs_per_eval, loss_scale=1
)


