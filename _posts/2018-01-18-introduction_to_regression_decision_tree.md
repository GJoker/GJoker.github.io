---
layout: post
title: 回归树构建示例
date: 2018-01-18
tags: 机器学习
music-id: 31654478
---

<div align="center">
	<img src="/images/posts/regression/regression.jpg" height="300" width="300">
</div>


Bengion 镇楼！

## 背景介绍

前段时间在空闲之余做了下 Kaggle 和天池的比赛，所以经常用到一些集成算法，像 stacking 和 boosting 等。因此，接下来几篇博文将会简单介绍下 boosting 的基本原理以及其经典算法。

用过 boosting 的人肯定知道，boosting 最常见的基分类器是决策树，特别的是 CART 树。CART 树的全称是 Classification and Regression Tree 即分类回归树。顾名思义，它既可以用于分类也可以用于回归预测。

其实关于 CART 树网上有很多资料，但奇怪的是基本上都是在讲如何构建分类树，而关于回归树的介绍却少之又少，所以我通过查找文献结合手头的资料，以示例的方式对如何构建回归树进行一个比较直白的讲解，希望有助于大家对于回归树的理解。

## CART回归树

### 概念介绍

我们知道决策树最为关键的点是在于选择什么样的划分策略来构建树模型。如经典的 ID3 树是通过信息增益来进行划分；C4.5 树是选择信息增益率来进行划分。而 CART 它首先假设整颗树是个二分类树，因此它的生成过程是递归地构建二叉树的过程。而它划分的准则视 CART 树的类别而定：如果是分类树则采用最小化基尼系数来选择特征进行划分；如果是回归树采用最小化平方误差来选择特征进行划分。关于分类树本文就不再赘述，可以自行去网上查阅文献，下文主要是讲解下回归树的构建。

### 构建流程

#### **算法流程**：

**输入**：训练集 D

**过程**：函数 TreeGenerate（D， A）:

1. 选择最优的切分属性 $j$ 和切分点 $i$ ，满足：

$$min_{j,i}[min_{c_1}\sum _{x_i \in R_1(j,i)} (y_i - c_1)^2+ min_{c_2}\sum _{x_i \in R_2(j,i)} (y_i - c_2)^2] \tag{1}$$

其中，$R_m$ 是被划分的输入空间，$y_i$ 表示样本真实的值，$c_m$ 表示空间 $R_m$ 对应的输出值，下文会介绍求解方法。

2. 用选定的 $(j, i)$ 对划分区域并计算各区域对应的输出值：

$$R_1(j,i) = \{x|x^{(j)} \leqslant i\}, \quad R_2(j,i) = \{x|x^{(j)} > i\} \tag{2}$$

$$c_m = \frac{1}{N_m} \sum_{x_k \in R_m(j,i)} y_k,   x\in R_m, m=1,2 \tag{3}$$

3. 对划分的子区域 $R_m$ 调用函数 TreeGenerate（$R_m$, A/{j}) 直至满足条件。

**输出**： 将输入空间划分成 M 个区域 $R_1, R_2, ..., R_M$ ,生成决策树：

$$f(x) = \sum_{m=1}^{M} c_m I(x \in R) \tag{4}$$

#### **示例**

上面那堆数学符号和数学公式理解起来确实比较头大，因此下面我通过示例来对上面的算法流程进行讲解。

因为只是为了比较直观地介绍回归树的构建过程，我就从最简单的数据入手，假设我们训练集只有一个特征进行划分，其具体内容如下：

|$x_i$|1|2|3|4|5|
|-|-|-|-|-|-|
|$y_i$|3.25|4.72|2.68|7.11|8.95|

根据流程，首先我们需要寻找一个合适的划分特征以及切分点使得公式 (1) 的值最小。由于本例中只有一个特征 $x$ ，所以我们把问题简化成了寻找一个划分点使得上式的值最小。

我们将 $x$ 按从小到大的顺序进行排列（上面已经排好）,根据所给的数据，考虑如下的划分点：
$$1.5,2.5,3.5,4.5$$

针对上述划分点，根据公式 (1) (2) (3) 我们不难求出不同划分点对应的 $R_1$、$R_2$、$c_1$、$c_2$ 以及：

$$m(s) = min_{j,i}[min_{c_1}\sum _{x_i \in R_1(j,i)} (y_i - c_1)^2+ min_{c_2}\sum _{x_i \in R_2(j,i)} (y_i - c_2)^2] $$

例如： 当 $i = 1.5$ 时，$R_1 = \{1\}$，$R_2=\{2,3,4,5\}$，$c_1=3.25$，$c_2=5.865$

$$m(s) = min_{j,i}[min_{c_1}\sum _{x_i \in R_1(j,i)} (y_i - c_1)^2+ min_{c_2}\sum _{x_i \in R_2(j,i)} (y_i - c_2)^2] = 0 + 22.52=22.52$$

以此类推，则可以得到如下结果：

| $i$ | 1.5 | 2.5 |3.5|4.5|
|:---:|-|-|-|-|
| $m(s)$  | 22.52  | 21.85  | 3.91  | 11.72 |

由上表可知，当划分点为 3.5 时 $m(s)$ 的值最小，此时 $R_1=\{1,2,3\}$，$R_2=\{4, 5\}$，$c_1=3.55$，$c_2=8.03$，因此可以得到回归树:
$$T_1(x)=
\begin{cases}
& 3.55, & x \leqslant 3.5 \\
& 8.03, & x > 3.5
\end{cases}
$$

$$f_1(x) = T_1(x)$$

接着，用 $f_1(x)$ 拟合训练数据的残差如下，表中 $y_i$ 的值为 $y_i - f_1(x_i), \quad i=1,2,...,10$

| $x_1$ | 1 | 2 | 3 | 4 | 5 |
|-|-|-|-|-|-|
| $y_i$  | -0.3  | 1.17  | -0.87  | -0.92  | 0.92  |

用 $f_1(x)$ 拟合训练数据的平方误差为：

$$L(y, f_1(x)) = \sum_{i=1}^{5}(y_i - f_1(x))^2 = 3.91$$

第二步求 $T_2(x)$，其具体的计算方法与求 $T_1(x)$ 完全一样，只是此时的训练数据变为了上面的残差，经过计算可以得到：

$$T_2(x)=
\begin{cases}
& -0.23, & x \leqslant 4.5 \\
& 0.92, & x > 4.5
\end{cases}
$$

$$f_2(x) = f_1(x) + T_2(x) =
\begin{cases}
& 3.32, & x\leqslant 3.5\\
& 7.8, & 3.5 < x \leqslant 4.5\\
& 8.95, & x > 4.5
\end{cases}
$$

用 $f_2(x)$ 拟合训练数据的平方误差为：
$$L(y, f_2(x))=\sum_{i=1}^{5}(y_i - f_2(2))^2=2.8506$$

以此类推，继续求得：
$$T_3(x) =
\begin{cases}
& 0.665, &  x \leqslant 2.5 \\
& 0.0167, & x>2.5
\end{cases}
$$

$$f_3(x) = f_x(x) + T_3(x) =
\begin{cases}
& 3.985 & x \leqslant 2.5 \\
& 3.337 & 2.5 < x \leqslant 3.5 \\
& 7.817 & 3.5 < x \leqslant 4.5 \\
& 8.967 & x > 4.5
\end{cases}
\quad L(y, f3(x)) = 1.97
$$

假设此时，满足设定的误差要求，那么 $f(x) = f_3(x)$ 就是所求的回归树。

## 总结

多特征的回归树构建和上面基本相同，只是会增加不同特征划分过程的计算，然后根据损失值选出最优的特征及最优划分点。至此回归树的构建就介绍完了，希望对大家有所帮助。
