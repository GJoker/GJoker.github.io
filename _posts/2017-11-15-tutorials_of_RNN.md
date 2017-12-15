---
layout: post
title: RNN 数学推导
date: 2017-11-15
tags: 数学杂谈
music-id: 31654478
---

## 简要介绍

　　Recurrent neural networks(RNNs) 是一系列功能强大且表征能力极强的网络结构，它们通常用于处理序列化的数据，目前 RNNs 在自然语言处理、语音识别上都取得了巨大的成功。

　　通常来说，RNNs 是以一组向量序列 $x_1$, $x_2$, ... , $x_n$ 作为其输入，然后根据不同的任务需求对这组向量序列一个接一个地进行处理。对于每一个新的输入数据 $x_i$，RNNs 都会更新其记忆存储并且得到一个新的隐含状态 $h_i$，你可以把 $h_i$ 看作是 RNNs 对序列 $x_1$ 到 $x_i$ 所有信息的记忆。因此，RNNs 的一个关键点是如何对 $h_i$ 进行更新，不同的更新方法通常对应一个不同的 RNN 网络结构。对于最简单的 RNN 它的更新公式如下所示：

$$h_t = f(x_t, h_{t-1}) \tag1$$

在上式中，$f$ 是一个抽象的函数，在给定 $t$ 时刻的输入 $x_t$ 以及 $t-1$ 时刻的状态 $h_{t-1}$，它能够计算出当前时刻的隐含状态 $h_t$ 。一个比较常见的 $f$ 形式如下：

$$h_t = \sigma(W_{xh}x_t + W_{hh}h_{t-1}) \tag2$$

其中，$\sigma$ 表示一个非线性函数如 sigmoid 或者 tanh，$W_{xh}$ 、$W_{hh}$ 分别表示连接输入和隐层以及连接隐层与隐层之间的权重矩阵。

对于每个时刻 $t$，RNNs 都可以(非必须，根据业务需求)得到一个输出值 $y_t$，这个值可为离散，也可以为连续值。在语言模型中该值通常用离散值表示，因为每一个离散值可以对应一个特定的字符。因此在离散的情形下，对于所有的输出类别，我们可以用一个概率分布 $p$ 进行表示：

$$s_t = W_{hy}h_t \tag3$$
$$p_t = \text{softmax}(s_t) \tag4$$

其中，$s_t$ 表示得分向量，也就是我们常说的 logits ，$W_{hy}$ 是连接隐含层和输出的权重矩阵。

对于每一个特定的类别 $y \in Y$，(4) 式可以写成：

$$p_t(y) = \frac{e^{s_t(y)}}{\sum_{y'\in Y}e^{s_t(y')}} \tag5$$

如下图所示，给出了一个 RNN 的简单示意图。

<div align="center">
	<img src="/images/posts/RNN/RNN.png" height="300" width="550">
</div>


## 网络训练和反向传播

　　对于语言模型而言，当给定一个训练集，假设它有 N 个离散的输出序列 $y^{(1)}$, ... , $y^{(N)}$，并且每个序列的长度为 $m_1$, ... , $m_N$。则可以将最终的目标函数定义如下：

$$
\begin{align*}
J(\theta) & = \sum_{i=1}^N -\log p(y^{(i)}) \\
          & = \sum_{i=1}^N \sum_{t=1}^{m_i} -\log p(y_t^{(i)} | y_{< t}^{(i)}) \tag6
\end{align*}
$$

### 数学工具
　　在对 RNN 反向传播进行推导之前，先科普几个数学推导式，便于理解后续梯度的求解(其实也就是链式法则)。

　　结论1：假设 $l$ 为损失值，已知其关于 $v$ 的梯度为 $dv$。假设 $v=f(Wh)$，则 $l$ 关于 $h$ 和 $W$ 的梯度分别为：

$$dh=W^{\text{T}} \cdot (f'(Wh) \circ dv) \tag7$$
$$dW = (f'(Wh) \circ dv) \cdot h^{\text{T}} \tag8$$

　　结论2：假设 $u$，$v$，$s$ 是任意的向量，并且 $s = u \circ f(v)$，则 $l$ 分别对 $u$，$v$，$s$ 的梯度为：
$$du = f(v) \circ ds \tag9$$
$$dv = f'(v) \circ du \circ ds \tag{10}$$

### 单时间戳反向传播

　　在推导整个序列的反向传播之前，我们首先推导一下单个时刻反向传播过程，即损失值 $l_t = -\log p_t(y_t)$ 关于变量 $\{W_{xh},W_{hh},W_{hy},x_t,h_{t-1}\}$ 的梯度求解过程。假设它们的梯度分别用符号 $\{dW_{xh},dW_{hh},dW_{hy},dx_t,dh_{t-1}\}$ 来表示，并根据公式 (3) 和 公式 (4) 定义中间梯度 $ds_t$，$dh_t$ 。

由公式 (5) 知 $p_t(y) = \frac{e^{s_t(y)}}{\sum_{y'\in Y}e^{s_t(y')}}$，则：

$$ds_t = \frac{\partial l_t}{\partial s_t}=\frac{\partial}{\partial s_t}(\log \sum_{y'}e^{s_t(y')}-s_t(y_t)) \tag{11}$$

对 (11) 式展开有：

$$\frac{\partial}{\partial s_t(y)}(\log \sum_{y'}e^{s_t(y')}-s_t(y_t))=\left\{\begin{matrix}
p_t(y_t) -1 & y=y_t \\
p_t(y) & y \neq y_t
\end{matrix}\right. \tag{12}$$

然后由公式 (3) 可知：

$$dh_t = W_{hy}^{\text{T}} \cdot ds_t \tag{13}$$
$$dW_{hy}=ds_t \cdot h_t^{\text{T}} \tag{14}$$
