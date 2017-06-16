---
layout: post
title: 使用 Tensorflow 实现 VGGNet
date: 2017-06-14
tags: Tensorflow
music-id: 530469
---

　　

<div align="center">
	<img src="/images/posts/tfimg/logo.jpg" height="300" width="500">
</div>

### VGGNet 简介
　　今天主要给大家介绍下 [VGGNet](https://arxiv.org/pdf/1409.1556/)，有过图像处理经验的人对 VGG 应该不会陌生，因为它的模型具有很强的迁移能力，以其为基础的 Pre-training model 被广泛地应用到了图像处理的很多领域中。

　　VGGNet 是由 Oxford 的 Visual Geometry Group 研发的深度卷积神经网络。它探索了卷积神经网络的深度与性能之间的关系，通过利用 `3×3` 的卷积核以及 `2×2` 的最大池化层进行反复堆砌，构筑了 16-19 层深的卷积神经网络(当时而言算是比较深的了)。并且它在 2014 年 ImageNet 的比赛中取得了分类任务的第 2 名和定位任务的第 1 名的成绩，与之前的 state-of-the-art 的网络结构相比，它的错误率有了大幅度下降。

　　在我将要介绍的四个经典的神经网络当中，我个人非常喜欢 VGG 模型，因为它非常地简洁高效。更重要的是对于我这种轻微的强迫症患者而言，统一大小的卷积和与池化层简直爽得不要不要的。

　　在 VGGNet 中，作者总结了以下几个观点：

- `LRN` 层的作用不明显。
- 网络结构越深效果越好。
- `1×1` 的卷积核很有效，但是效果没有 `3×3` 的卷积效果好，大一点的卷积核能够学习到比较大的空间特征。

### VGGNet 网络结构

　　VGGNet 全部都是使用的 `3×3` 的卷积核和 `2×2` 的池化核，为了探索深度对网络性能的影响，设计了从 A～E 5个级别的网络模型框架，如下图所示。

<div align="center">
	<img src="/images/posts/vggnet/vggnet.png" height="550" width="400">
</div>

　　从上图可以看出从 A 到 E 每个级别的网络的层数逐渐加深，但是网络的参数量并未增长很多。这是因为，网络中的参数主要集中在最后的全连接层，前面的卷积层虽然很深，但是其参数的消耗量比不是很大。

　　VGGNet 包含有5个卷积段，每个卷积段包含有 2-4 个卷积层。相同卷积段中的卷积层包含的卷积核的个数相同，越靠后的卷积段的卷积核个数越多，分别为64、128、256、512、512。然后在每个卷积段的末尾都会接一个最大池化层以减小图片的尺寸。

　　前面介绍过了，VGGNet 大部分都是使用的 `3×3` 卷积核堆砌的而成的，相比于 AlexNet 中使用的 `11×11` 、`7×7` 和 `5×5` 卷积核，`3×3` 卷积核有什么优势呢？首先看下图。

<div align="center">
	<img src="/images/posts/vggnet/kenel.jpg" height="450" width="350">
</div>

从图中我们可以看到两个 `3×3` 卷积核堆砌就相当于一个 `5×5` 的卷积核效果，以此类推，一个 `7×7` 相当于3个 `3×3` 进行串联。然而我们发现两个 `3×3` 的卷积核的参数数量是18，而一个 `5×5` 的卷积核的参数数量却是25，反而还减少了。与此同时前者还做了两次非线性变换(后者只有一次)，使得网络对特征的学习能力更强了。因此文中都是用 `3×3` 的卷积进行串联而成。

### VGGNet 的实现

　　介绍了一堆 VGG 相关的内容，废话不多说到了我们的实现环节。总体而言 VGGNet 的结构还是非常简洁明了的，下面主要是利用 Tensorflow 对 VGGNet-16 进行代码实现。

　　首先，定义几个相关的 op 操作，供后面构建网络时使用：

```python
from datetime import datetime
import math
import time
import tensorflow as tf

NUM_BATCHES = 32
BATCH_SIZE = 100

# 定义卷积操作
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
  # kh、kw分别表示卷积核的高和宽，dh、dw分别表示移动的步长
  n_in = input_op.get_shape()[-1].value

  with tf.name_scope(name) as scope:
    kernel = tf.get_variable(scope + 'w', shape=[kh, kw, n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())

    conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
    bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
    biases = tf.Variable(bias_init_val, trainable=True, name='b')
    z = tf.nn.bias_add(conv, biases)
    activation = tf.nn.relu(z, name=scope)
    p += [kernel, biases]
    return activation

# 定义全连接操作
def fc_op(input_op, name, n_out, p):
  n_in = input_op.get_shape()[-1].value

  with tf.name_scope(name) as scope:
    kernel = tf.get_variable(scope + 'w', shape=[n_in, n_out], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
    activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
    p += [kernel, biases]
    return activation

# 定义最大池化操作
def mpool_op(input_op, name, kh, kw, dh, dw):
  return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)
```

　　完成上述函数的定义之后，接下来接开始构建 VGGNet 的网络结构。可以将其看成由六段结构组成，其中前五段为卷积段，最后一段的是全连接段。根据前面所示的 VGGNet-16 的网络结构图我们构建如下：

```python
def inference_op(input_op, keep_prob):
  p = []

  # 第一个卷积段
  conv1_1 = conv_op(input_op, name='conv1_1', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
  conv1_2 = conv_op(conv1_1, name='conv1_2', kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
  pool1 = mpool_op(conv1_2, name='pool1', kh=2, kw=2, dw=2, dh=2)

  # 第二个卷积段
  conv2_1 = conv_op(pool1, name='conv2_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
  conv2_2 = conv_op(conv2_1, name='conv2_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
  pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

  # # 第三个卷积段
  conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
  conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
  conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
  pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)

  # 第四个卷积段
  conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
  conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
  conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
  pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

  # 第五个卷积段
  conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
  conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
  conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
  pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

  shp = pool5.get_shape()
  flattened_shape = shp[1].value * shp[2].value * shp[3].value
  resh1 = tf.reshape(pool5, [-1, flattened_shape], name='resh1')

  # 第一个全连接层
  fc6 = fc_op(resh1, name='fc6', n_out=4096, p=p)
  fc6_drop = tf.nn.dropout(fc6, keep_prob, name='fc6_drop')

  # 第二个全连接层
  fc7 = fc_op(fc6_drop, name='fc7', n_out=4096, p=p)
  fc7_drop = tf.nn.dropout(fc7, keep_prob, name='fc7_drop')

  # 第三个全连接层
  fc8 = fc_op(fc7_drop, name='fc8', n_out=1000, p=p)
  softmax = tf.nn.softmax(fc8)
  predictions = tf.argmax(softmax, 1)
  return predictions, softmax, fc8, p
```

　　然后定义评测函数 `time_tensorflow_run()` 对网络的运算性能进行评测，并定义主函数进行测试。

```python
def time_tensorflow_run(session, target, feed, info_string):
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in range(NUM_BATCHES + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target, feed_dict=feed)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration

  mn = total_duration / NUM_BATCHES
  vr = total_duration_squared / NUM_BATCHES - mn * mn
  sd = math.sqrt(vr)
  print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, NUM_BATCHES, mn, sd))

def main():
  with tf.Graph().as_default():
    image_size = 224
    images = tf.Variable(tf.random_normal([BATCH_SIZE, image_size, image_size, 3], dtype=tf.float32, stddev=0.1))

    keep_prob = tf.placeholder(tf.float32)
    predictions, softmax, fc8, p = inference_op(images, keep_prob)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)

      time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
      objective = tf.nn.l2_loss(fc8)
      grad = tf.gradients(objective, p)
      time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")

if __name__ == "__main__":
  main()
```
### 总结
　　VGGNet 在 ILSVRC 2014 的比赛中最终得到了 7.3% 的错误率，相比 AlexNet 取得很大的进步，并且凭借其简洁高效的网络结果以及优秀的分类能力，被应用到了许多地方，其很多思想都值得我们学习和借鉴。
