---
layout: post
title: MCＭC 方法介绍
date: 2017-06-08
tags: 数学杂谈
music-id: 784594
---



　　记得在当初学习 **受限制波尔兹曼机(Restricted Boltzmann Machine, RBM)** 的时候第一次接触到了 **马尔科夫链蒙特卡罗方法(Markov chain Monte Carlo，MCMC)** ，当时对其原理简直就是一脸懵逼，果断放弃继续深究。后来在研究 **generative models** 的时候，在很多的 Paper 里再次看到这个关键词。于是就抱着我不入地狱谁入地狱的心态，查阅各种文献资料，终于对其原理有了个初步了解，于是觉得很有必要将其整理成学习笔记。在讲 **MCMC** 之前，先简单地介绍下**蒙特卡罗方法**。

<div align="center">
	<img src="/images/posts/tfimg/minguo2.jpg" height="100" width="100">
</div>

### 蒙特卡罗方法
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

其中，$\mathbf{P}=(P_{i,j})_{n×n}$为**转移概率矩阵**
　　思考一个问题，如果存在某个状态，从它出发转移回自身所需的转移次数总是整数 $d$ 的倍数,因此这个马尔科夫过程就具有周期性。如果任意两个取值之间总是能以非零的概率进行相互转移，那么这个马尔科夫过程称为不可约——“不可约”是指每一个状态都可以来自任意的其它状态。假如一个马尔科夫过程既没有周期性，又不可约。则称其为**各态遍历的**
　　对于各态遍历的马尔科夫过程，不论 $\pi^{(0)}$ 取何值，随着转移次数的增多，随机变量的取值分布最终都会收敛于**唯一的**平稳分布 $\pi^{*}$ ，即

$$\lim_{t\to \infty} \pi^{(0)} \mathbf{P}^t= \pi^{*} \tag{9}$$

并且这个平稳分布 $\pi^{*}$ 满足

$$\pi^{*} \mathbf {P} = \pi^{*} \tag{10}$$

于是，不论 $\pi^{0}$ 取何值，经过足够多次的转移后，随机变量各取值总会不断接近于该过程的平稳分布。这便是 MCMC 的理论基础：如果希望在某个分布下进行采样，只需要模拟以其平稳分布的马尔科夫过程，经过足够多次转移后，我们的样本分布就会充分接近于该平稳分布。便意味着我们近似地采集到了目标分布下的样本。

### 正则分布
　　假设一个物理系统具备一定的自由度，如一滴水珠中的分子在空间可以任意地排列，那么，系统所处的状态(所有分子的位置)就具备一定的随机性。假设系统处于状态 $i$ 的概率为 $p_i$ ，根据系统的物理性质，不同的状态可能会使系统具备不同的能量，如果用 $E_i$ 表示系统处于状态 $i$ 时的能量，在统计力学中有一个基本结论:当系统于外界达到热平衡时，系统处于状态 $i$ 的概率 $p_i$ 具有以下的形式

$$p_i = \frac{1}{Z_T}e^{-\frac{E_i}{T}} \tag{11}$$

其中:

$$Z_T = \sum_i e^{-\frac{E_i}{T}} \tag{12}$$

被称为归一化函数，这种概率分布的形式叫做正则分布，其实机器学习中常见的 softmax 函数就是一个正则分布。

　　从分布的形式可以看出，能量越小的状态具有越大的概率。在机器学习领域，人们通常根据需求自定能量函数，然后借鉴物理规律去实现其他功能。

### Metropolis-Hastings 采样
　　在 MCMC 算法中，我要在一个指定的分布上进行采样，首先从系统的任意状态出发，模拟马尔科夫过程，不断地进行状态转移。根据上面所说的马尔科夫的性质，在经过足够的转移次数之后，我们所处的状态即符合目标分布，这时，该状态就可以作为一个采集到的样本。于是算法的关键在于设计一个合理的状态转移过程，Metropolis-Hastings 是一个非常重要的 MCMC 采样算法。

　　假设我们想要从分布 $\pi(\cdot)$ 上采集样本，于是，我们需要引入另一组概率分布，将之称为转移提议分布(proposal density) $Q(.;i)$。这个分布的作用是根据当前的状态 $i$ 提议转移之后的状态。每次转移时，我们首先利用 $Q(.;i)$ 提议出下一步的状态，假设为 $j$ ，然后，以下面的概率接受这个状态

$$\alpha(i \to j)=\min {\{1, \frac{\pi(j)Q(i;j)}{\pi(i)Q(j;i)}\}} \tag{13}$$

上式中，$\alpha(i \to j)$ 被称作接受概率。

　　为了模拟接受新状态 $j$ 的过程，可以首先产生一个 $[0,1]$ 之间的均匀分布的随机数 $r$ 然后，如果 $r<\alpha(i \to j)$, 则采用状态 $j$ 作为新状态，否则维持状态 $i$ 不变，$Q(j;i)$ 表示从状态 $i$ 提议转移到状态 $j$ 的概率。一般来说，$Q(.,.)$ 可以选择一些比较简单的概率分布。

　　下面简单推导下 Metropolis-Hastings 算法。

　　首先，根据上面的介绍，状态 $i$ 转移到状态 $j$ 的转移概率为:

$$P(i \to j)=\alpha(i \to j)Q(j;i \tag{14})$$

于是我们很容易知道，对于任意的状态 $i,j$ 下式成立:

$$
\begin{align*}
\pi(i) P(i \to j) & = \pi(i)\alpha(i \to j)Q(j;i)  \\
& = \pi(i) \min {\{1, \frac{\pi(j)Q(i;j)}{\pi(i)Q(j;i)}\}}Q(j;i)  \\
& = \min {\pi(i) Q(j;i), \pi(j)Q(i;j)} \\
& = \pi(j) \min {\{1, \frac{\pi(i)Q(j;i)}{\pi(j)Q(i;j)}\}}Q(i;j)  \\
& = \pi(j)\alpha(j \to i)Q(i;j) \\
& = \pi(j) P(j \to i) \tag{15}
\end{align*}
$$

于是可以证明状态分布 $\pi(\cdot)$ 在转移概率 $P(i \to j)$ 下保持不变，即:

$$
\sum_j \pi(j)P(j \to i) = \sum_j \pi(i)P(i \to j)=\pi(i)\sum_jP(i \to j)=\pi(i) \tag{16}
$$

如果该马尔科夫过程满足各态遍历性，那么，根据稳定分布的唯一性，我们就知道在转移 $P$ 下，状态的分布最终收敛于 $\pi(\cdot)$，也就是我们要采样的分布。

### Gibbs 采样
　　Gibbs 采样是 Metropolis-Hastings 采样的特殊形式，它应用于系统具有多个变量，并且对于变量间的条件分布我们能够直接采样的情况下。在 Gibbs 采样中，假设系统由 $m$ 个变量构成，定义系统状态 $X=(x_1,x_2,\dots,x_m)$ 并且对于任一变量 $x_i$ ，我们能够直接从条件分布 $P(x_i|x_i,\dots,x_{i-1},x_{i+1},\dots,x_m)$ 中为其采样。

　　关于 Gibbs 采样具体流程以后会详细介绍，这次就不再啰唆了。
