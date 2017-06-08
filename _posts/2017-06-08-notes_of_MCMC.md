---
layout: post
title: 马尔科夫链蒙特卡罗(Markov chain Monte Carlo，MCMC)方法
date: 2017-06-08
tags: 数学杂谈
music-id: 407000263
---

　　记得在当初学习 **受限制波尔兹曼机(Restricted Boltzmann Machine, RBM)** 的时候第一次接触到了 **MCMC方法** ，当时对其原理简直就是一脸懵逼，果断放弃继续深究。后来在研究 **generative models** 的时候，在很多的 Paper 里再次看到这个关键词。于是就抱着我不入地狱谁入地狱的心态，查阅各种文献资料，终于对其原理有了个初步了解，于是觉得很有必要将其整理成学习笔记。

## MCMC 方法介绍
### 蒙特卡罗方法
　　在讲 **MCMC** 之前，先简单地介绍下**蒙特卡罗方法**。
　　最早的**蒙特卡罗方法**是应用在物理上的，它主要是用于通过随机化的方法计算积分。如给定函数 $h(x)$ ，思考计算如下积分：
$$\int_{a}^{b}h(x)dx \tag{1}$$
　　很多情况下我们无法使用数学推导直接求出解，大牛们想了很多方法来逼近真实解，**蒙特卡罗方法** 就是其中一个比较著名的方法。它的大致想法是我们可以将 $h(x)$ 分解成某个函数 $f(x)$ 以及一个定义在区间 $(a,b)$ 上的概率密度函数 $p(x)$ 的乘积，于是公式 $(1)$ 就可以写成下面的形式：
$$\int_{a}^{b}h(x)dx=\int_{a}^{b}p(x)f(x)dx=\mathbb{E}_{p(x)}[f(x)] \tag{2}$$
　　于是原来的积分就等价于 $f(x)$ 在 $p(x)$ 这个分布上的期望了，那么我们便可以通过从分布 $p(x)$ 上采集大量的样本点 $x_1,x_2,\dots,x_n$ ，即对于 $\forall{i}$ 有：
$$\frac{x_i}{\sum_{i=1}^n x_i} \approx p(x_i)$$
那么便可以通过这些的样本去逼近这个期望
$$\int_a^b h(x)dx = \mathbb{E}_{p(x)}[f(x)] \approx \frac{1}{n} \sum_{i=1}^n f(x_i) \tag{3}$$
　　上面便是蒙特卡罗方法的基本思想。其实可以很容易看到蒙特卡罗方法的一个关键性问题便是：如何在一个定义好的分布 $p(x)$ 上进行采样。对于比较经典的分布像高斯分布、均匀分布等目前已有了成熟的算法快速地生成无偏样本。但是对于任意的分布，还无法做到这一点。于是**马尔科夫链蒙特卡罗方法**就是来解决任意分布下的采样问题的。
### 马尔科夫链
　　MCMC 的基本思想是利用马尔科夫链来产生指定分布下的样本，因此，先简单的回顾下马尔科夫链的基本知识。

　　假设 $X_t$ 表示随机变量 $X$ 在时间 $t$ 的取值，马尔科夫过程是指如果该变量随时间变化的转移概率仅仅依赖于它当前时刻的状态，即：

$$P(X_{t+1}=s_j | X_0 = s_{i0}, X_1=s_{i1},\dots,X_t=s_i)=P(X_{t+1}=s_j|X_t=s_i) \tag{4}$$

其中，$s_{i0},s_{i1},\dots,s_{i},s_{j}$ 为随机变量 $X$ 可能的状态。

　　从上式可以看出，对于一个马尔科夫变量，如果我们知道了当前时刻的取值，我们便可以对其下一个时刻的值进行预测。所谓的马尔科夫链是指一段时间内随机变量 $X$ 的取值序列 $(X_0,X_1,\dots,X_m)$ 符合公式(4)。

　　一个马尔科夫链通常可以通过其对应的转移概率来定义，所谓的转移概率是指随机变量从一个时刻到下一个时刻，从状态 $s_i$ 转移到另一个状态 $s_j$ 的概率：

$$P(i \rightarrow j):= P_{i,j}= P(X_{t+1}=s_j|X_t=s_i) \tag{5}$$

如果用 $\pi_{k}^{(t)}$ 表示随机变量 $X$ 在时刻 $t$ 取值 $s_k$ 的概率，则 $X$ 在时刻 $t+1$ 取值为 $s_i$ 的概率为：


$$
\begin{align*}
 \pi_{i}^{(t+1)} & = P(X_{t+1}=s_i) \\
& = \sum_k P(X_{t+1}=s_i | X_t = s_k) \cdot P(X_t = s_k) \\
& = \sum_k P_{k,i} \cdot \pi_k^{(t)} \tag{6}
\end{align*}
$$

　　假设一共有 $n$ 种状态，则根据根据公式 $(6)$ 有：

$$
(\pi_1^{(t+1)},\dots,\pi_n^{(t+1)})=(\pi_1^{(t)},\dots,\pi_n^{(t)}) \begin{bmatrix}
 P_{1,1}&  P_{1,2}&  \cdots & P_{1,n}\\
 P_{2,1}&  P_{2,2}&  \cdots & P_{2,n}\\
 &  &  \cdots & \\
 P_{n,1}&  P_{n,2}&  \cdots & P_{n,n} \tag{7}
\end{bmatrix}
$$

用矩阵形式可以表示为：
$$\pi^{(t+1)} = \pi ^{t} \cdot \mathbf{P} \tag{8}$$
其中，$\pi^{(t)}$
　　思考一个问题，如果存在某个状态，从它出发转移回自身所需的转移次数总是整数 $d$ 的倍数