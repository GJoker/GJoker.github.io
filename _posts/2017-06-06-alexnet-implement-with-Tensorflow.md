---
layout: post
title: 使用 Tensorflow 实现 AlexNet
date: 2017-06-06
tags: Tensorflow
music-id: 407000263
---

最近这段时间将会更新几篇博客，来简单地介绍下几款经典的卷积神经网络模型，包括：AlexNet、VGGNet、Google IncepNet、ResNet。并使用 Tensorflow 分别对其进行代码实现。

<div align="center">
	<img src="/images/posts/tfimg/logo.jpg" height="300" width="500">
</div>


### AlexNet 简介
AlexNet 是由 Hinton 的学生 Alex Krizhevsky 在2012年提出来的。并在 ILSVRC 2012 比赛中以显著的优势取得当年的冠军，它将top-5的错误了降低到了16.4%，和第二名的26.2%相比，成绩有了质的飞跃。AlexNet 可以说是神经网络在其低谷时期后的第一次发声，迅速确立了深度学习(深度卷积神经网络)在计算机视觉中的统治地位。与此同时，也推动了深度学习在自然语言处理、语音识别已经强化学习等领域的发展。

当年的 AlexNet 我们现在回过头来看其结构其实是非常简单的，它主要的技术特点在于以下几个方面：

- 使用了 Relu 作为 CNN 的激活函数，替换掉了过去经常使用的 Sigmoid 函数， 解决了 Sigmoid 在网络较深的时候会出现的梯度消失问题。
- 在训练的时候使用了 Dropout 随机忽略一部分神经元，用来避免过拟合，提高模型的泛化能力。
- 使用了重叠的最大池化，避免了平均池化的模糊化问题，同时使步长比池化核的尺寸小，一定程度上提升了特征的丰富性。
- 提出了 LRN 层，使得其中相应比较大的值变得更大，并抑制了其他反馈比较小的神经元，类似于正则项的功能，一定程度上提高了模型的泛化能力。
- 使用了 CUDA 加速深度神经网络的训练。通过GPU来处理神经网络训练中大量的矩阵运算，大大地提高了训练速率。
- 数据增强。所谓的数据增强是指作者将原有的数据集进行了相关变换，大大地增加了数据量。使其减轻了过拟合的风险。

### AlexNet 网络结构
整个 AlexNet 网络模型如下图所示。从图中可以看到 AlexNet 其实是一个8层的网络结构，其中前5层是卷积层，后面的3层是全连接层。在第1个和第2个卷积层之后使用了 LRN 层，而 Max pooling 则使用在两个 LRN 层以及最后一个卷积层后。这8层结构每层都使用 Relu 作为其激活函数。

<div align="center">
	<img src="/images/posts/tfimg/alexnet.png" height="200" width="550">
</div>


其中 AlexNet 的超参数如下图所示。从图中可以发现一个很有意思的现象，虽然卷积层的计算量虽然非常大，但是参数量却非常的小，最多的一层也才1.3M，其余的才几百K而已，只占了 AlexNet 总参数量中很小的一部分。这就是卷积的优势所在，能够利用较少的参数量来提取有效的特征。每个卷积层所包含的参数量连整个网络的层参数的1%都不到，但是如果去掉其中任何一个卷积层，都会使得网络的性能大幅度下降

<div align="center">
	<img src="/images/posts/tfimg/alexparams.png" height="350" width="180">
</div>


### AlexNet 实现

本文对 AlexNet 实现将不涉及实际数据的训练，只是构建一个完成 AlexNet 的网络模型，然后测试下网络的运行速率。有兴趣的童鞋可以使用该代码对真实的数据集进行测试，看看结果如何。

首先，导入几个会用到的库，并且定义一个 `print_activations()` 函数，用于显示输入图片经过每层网络处理后得到输出 Tensor 的尺寸大小。

```python
from datatime import datatime
import time
import math
import tensorflow as tf

BATCH_SIZE = 64
NUM_BATCHES = 100

def print_activations(t):
  print(t.op.name, '', t.get_shape().as_list())
```

然后，定义 `inference()` 函数，即用于构建整个网络的框架，描述从接受图片作为输入开始并得到最终输出结果的前向传播过程。网络中的各个

```python
def inference(images):
  parameters = [] # 收集需要 training 的参数，用于后面的梯度计算。在真实的模型中是不需要的，因为tensorflow可以帮你自动求导，不需要你去手动计算。

  # 定义第一个卷积层
  with tf.name_scope('conv1') as scope:
    kernel = tf.get_variable(name='weights', shape=[11, 11, 3, 64], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[64], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

  lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
  pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
  print_activations(pool1)

  # 定义第二个卷积层
  with tf.name_scope('conv2') as scope:
    kernel = tf.get_variable(name='weights', shape=[5, 5, 64, 192], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[192], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    print_activations(conv2)
    parameters += [kernel, biases]

  lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
  pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
  print_activations(pool2)

  # 定义第三个卷积层
  with tf.name_scope('conv3') as scope:
    kernel = tf.get_variable(name='weights', shape=[3, 3, 192, 384], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[384], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    print_activations(conv3)
    parameters += [kernel, biases]

  # 定义第四个卷积层
  with tf.name_scope('conv4') as scope:
    kernel = tf.get_variable(name='weights', shape=[3, 3, 384, 256], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    print_activations(conv4)
    parameters += [kernel, biases]

  # 定义第五个卷积层
  with tf.name_scope('conv5') as scope:
    kernel = tf.get_variable(name='weights', shape=[3, 3, 256, 256], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.get_variable(name='biases', shape=[256], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    print_activations(conv5)
    parameters += [kernel, biases]

  pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')
  print_activations(pool5)

  return pool5, parameters

```

然后定义一个用于评估每轮计算时间的函数 `time_tensorflow_run()`。其中参数 `num_steps_burn_in=10` 表示预热轮数，可以看成是在给程序热身，因为最开始的几轮要面临显存加载、cache命中等问题，所得到的时间并不准确因此我们可以跳过，只需考虑10轮迭代之后的计算时间。

```python
def time_tensorflow_run(session, target, info_string):
  num_step_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0

  for i in range(NUM_BATCHES + num_step_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i >= num_step_burn_in:
      if not i % 10:
        print('%s : step %d, duration = %.3f' % (datetime.now(), i - num_step_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration

  mn = total_duration / NUM_BATCHES
  vr = total_duration_squared / NUM_BATCHES - mn * mn
  sd = math.sqrt(vr)
  print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, NUM_BATCHES, mn, sd))

```

最后定义主函数如下：

```python
def main():
  with tf.Graph().as_default():
    image_size = 224
    images = tf.get_variable('input', shape=[BATCH_SIZE, image_size, image_size, 3], dtype=3, initializer=tf.random_normal_initializer(stddev=0.1))

    pool5, parameters = inference(images)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    time_tensorflow_run(sess, pool5, "Forward")

    objective = tf.nn.l2_loss(pool5)
    grad = tf.gradients(objective, parameters)
    time_tensorflow_run(sess, grad, "Forward-backward")

if __name__ == "__main__":
  main()
```

通过上面的代码，我们可以对 AlexNet 计算耗时进行测量。通常情况下 CNN 的训练过程是比较耗时间的，它需要遍历很多遍数据，进行大量的迭代工作。因此，想要应用 CNN 最关键的问题在于训练过程，模型训练好了之后预测过程可以看成是实时的。

### 小结

AlexNet 的出现为我们彻底打开深度学习的这扇大门，它取得的巨大突破一方面是所设计的科学合理的网络结构，另一方面则得益于海量的标注数据集(ImageNet)以及强大 GPU 并行计算能力。

当年 AlexNet 发表在 NIPS 时候，Hinton 曾说道“如果你没参加过之前十几年的 NIPS，那没关系，因为直到今年神经网络才真正开始生效。”字里行间无不流漏出自信和霸气。
