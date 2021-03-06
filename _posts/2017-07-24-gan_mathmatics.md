---
layout: post
title: Generative Adversarial Network(GAN) 中的数学思想
date: 2017-07-24
tags: 生成模型
music-id: 31654478
---

　　随着 GAN 越来越火，最近发现身边有越来越多的朋友开始尝试使用 GAN 作为生成模型，生成各种各样有趣的东西。大家在学习 GAN 的过程中，有很多人可能会有疑问，为什么 GAN 的这种架构就能使得生成的东西越来越逼真呢？它的数学原理是什么？今天，就给大家简单介绍下 GAN 中蕴含的数学思想。

<div align="center">
	<img src="/images/posts/GAN/logo.jpg" height="300" width="500">
</div>

### 生成模型简单介绍
　　Generative Adversarial Network，从 GAN 的名字就可以知道，它的核心就是生成。对于一个理想的生成模型，我们希望在给机器看过一堆 Object 之后，它能够学会自动地生成相似的东西。举例来讲，比如让机器看过一堆人脸图片，让它能够学会生成逼真的人脸图片；或者，给机器看唐诗三百首之后，让它能够自动生成一首的唐诗。

　　关于生成模型的研究，早在二十多年之前就有人提出来来过，但由于某些条件的限制，使得生成模型直到最近几年才开始快速发展。对于生成模型，其中一种我们比较熟悉是 Auto-encoder ，它的基本原理如下图所示：

<div align="center">
	<img src="/images/posts/GAN/AE.png" height="300" width="500">
</div>

　　首先，训练一个神经网络作为 Encoder 将原始图片编码成一个 code，然后，再训练一个新的网络作为 Decoder 将 code 重新转换成图片，使得它和原始图片越接近越好。于是可以看到，如果我们将训练好的整个网络中的 Encoder 部分去掉，剩下部分就可以看成一个简单的生成模型。但是常规的 AE 作为生成器的效果很差，生成的图片失真很严重。

　　于是人们在 AE 的基础上对其进行了改进，提出了一种名叫 Variational Auto-encoder(VAE 变分自编码) 的生成模型，它的工作原理如下图所示：

<div align="center">
	<img src="/images/posts/GAN/VAE.png" height="300" width="500">
</div>

　　从上图可以看出，VAE 其实是在原来的 AE 模型的基础上，对 code 部分进行了修改。将原来 Encoder 生成的 code 拆成了两部分，一部分作为均值，另一部分作为标准差，并将它们经过一些的变换得到一组新的 code ，然后和 AE 一样以此作为 Decoder 的输入尽可能地还原成原始的图片。

　　当然整个 VAE 在架构并没有这么简单，关于 VAE 的具体的思想，我会在后续的博文中进行介绍，这里就不再赘述。VAE 是在 GAN 出现之前最为有效的生成模型，但是 VAE 有个问题，就是它在判断一张图片是不是真实的方式是看生成的图片与原始的图片是不是接近的，然而是否是真实的和与真实图片是否接近有时候可能是不一样的概念。

　　举例来说，在下图中，两张生成的图片与真实的图片都只差了一个 pixel ，在人的眼中，左边的图片就是真实的，而右边的则不是。但是对于 VAE 而言，这两张图片的真实程度就是一样的。因此，VAE 生成的图片通常会比较模糊，效果不是特别理想。

<div align="center">
	<img src="/images/posts/GAN/pixel.png" height="100" width="500">
</div>

接下来我们就来介绍一下今天的主角 GAN ，它能够帮助我们生成更加真实的图片。

### GAN 的简单介绍

　　相比于生成模型 VAE 而言，GAN 的网络架构更加灵活，并且它的原理比较容易被大家接受，主要是因为在我们的日常生活中，有很多现象都与 GAN 的过程类似。就如 Ian Goodfellow 举例说的一样，GAN 中的判别器好比是一个验钞机，它要不断地提高自己的防伪能力，尽可能地找出假钞；而 GAN 中的生成器则像是一名假钞制造商，它希望自己制造的假钞尽可能地骗过验钞机。假钞制造商不断地对抗中提升自己的能力，最终，使得生成的假钞越来越逼真。

　　GAN 在训练的过程与上面的例子类似，可以把它看成是一个演化的过程，如下图所示：首先有一个第一代的生成器，它会生成一组图片，然后将这组生成的图片与真实的图片丢给判别器进行判断，让它来判断哪些是生成的哪些是真实的，根据得到的结果利用优化算法对判别器的参数进行修改，使得判别器能够尽可能地分辨出真实样本和第一代生成器生成的样本，从而得到第一代判别器。然后再根据第一代判别器对生成器进行训练更新得到第二代生成器，使其生成的样本能尽可能地使第一代判别器无法分辨哪些是生成的样本，哪些是真实的样本。然后以此类推，就像这样不断的进行迭代更新，最终使得生成器生成的样本越来越真实。

<div align="center">
	<img src="/images/posts/GAN/evolution.png" height="300" width="500">
</div>

### GAN 的数学原理
　　上面对 GAN 的原理有了一个比较抽象的认识，然而为什么上面介绍的训练过程能够使得生成的样本越来越逼真呢？它的数学依据是什么？为了回答这个问题，就需要对 GAN 的数学原理进行推导。

#### **最大似然估计**

　　假设有一个真实的数据分布 $P_{data}(x)$，但是并不知道这个分布的具体形式，于是希望有这么一个概率分布 $P_{G}(x; \theta)$，它由参数 $\theta$ 完全控制，能够尽可能地去拟合这个真实的数据分布，使得 $P_{G}(x; \theta)$ 与真实的分布 $P_{data}(x)$ 越接近越好。这样就可以利用 $P_{G}(x; \theta)$ 来生成与真实数据非常近似的样本了。

　　那么如何进行拟合呢，虽然无法知道真实样本分布的具体形式，但是可以对它进行采样。假设从真实的样本分布 $P_{data}(x)$ 中 sample 了 m 个样本 ${x^1, x^2, \dots, x^m}$，于是就可以计算出 $P_{G}(x^i; \theta)$ 的值，从而得到这组采样的似然函数：

$$L=\prod_{i=1}^{m} P_{G}(x^i; \theta) \tag{1}$$

　　既然这 m 个样本来自于真实的样本空间，那当然是希望 $P_{G}(x^i; \theta)$ 越大越好，所以我们的目的是寻找一组参数 $\theta^*$ 使得上式最大，这也就是最大似然估计的思想。如果用数学公式描述，并对它进行一些变换可以得到如下的推导：

$$
\begin{align*}
\theta^*   & = arg \max _{\theta} \prod _{i=1}^{m} P_{G}(x^i; \theta) = arg \max _{\theta} \log \prod _{i=1}^{m} P_{G}(x^i; \theta) \\
& = arg \max _{\theta} \sum _{i=1}^{m} \log P_{G}(x^i; \theta) \\
& \approx arg \max _{\theta} \mathbb{E} _{x \sim P_{data}} [\log P_{G}(x; \theta)] \\
& = arg \max _{\theta} \int _{x} P_{data}(x) \log P_{G}(x; \theta) dx \\
& = arg \max _{\theta} \int _{x} P_{data}(x) \log P_{G}(x; \theta) dx - \int _{x} P_{data}(x) \log P_{data}(x) dx \\
& = arg \max _{\theta} \int _{x} P_{data}(x) \log \frac {P_{G}(x; \theta)}{P_{data}(x)} dx \\
& = arg \min _{\theta} KL(P_{data}(x) || P_{G}(x; \theta)) \tag{2}\\
\end{align*}
$$

　　由公式(2)的变换结果可以看出，求解最大似然估计的过程实际上是等价于寻找一组参数 $\theta ^*$ ，使得两个分布 $P_{G}(x^i; \theta)$ 和 $P_{data}(x)$ 的 KL 散度最小，而我们知道 KL 散度越小代表两个分布越接近，所以这与我们的最终目的是一致的。

　　在了解原理之后，现在问题的关键在于怎样去定义一个的 $P_{G}(x; \theta)$ 去拟合原始的数据分布。你当然可以选择一些常见的分布比如高斯之类的去拟合，但问题在于真实的数据分布 $P_{data}(x)$ 通常会非常的复杂，它与这些分布有很大的不同，这样拟合得到的结果效果一般都比较不理想。因此，在 GAN 中选择使用神经网络进行拟合，因为在理论上，神经网络可以拟合出非常复杂的函数。

<div align="center">
	<img src="/images/posts/GAN/nn.png" height="300" width="600">
</div>

　　从上图可以看到，其具体思路是先选择一个先验分布 $P_{prior}(z)$，这个分布可以是任何一个你熟知的分布，比如高斯、均匀分布等。然后将这个分布作为输入丢入神经网络中，通过变换可以得到一个新的分布 $P_{G}(x; \theta)$，再通过调整参数 $\theta$ ，使其与 $P_{data}(x)$ 尽可能的接近。如果用数学公式描述可以写成以下形式：

$$P_{G}(x) = \int _{z} P_{prior}(z) I_{[G(z)=x]} dz \tag{3} $$

　　然而比较遗憾的是，上式我们无法将其写成似然函数的形式，因为 $G(z)$ 的具体形式无从知晓。于是就不能通过前面所说的最大似然估计的方法对参数进行迭代更新了。那么如何在无法计算似然函数的情况下，度量 $P_{G}(x)$ 与 $P_{data}(x)$ 之间的相似性，并通过调整参数使得 $P_{G}(x)$ 与 $P_{data}(x)$ 越来越接近呢？ GAN 就为我们提供一种求解思路，而它的最大贡献也就在于此。

#### **GAN 的基本理论**

　　下面简单地介绍下 GAN 所提出的解决思路：
- 首先，定义一个生成器 G，它的输入是 z，输出是 x，当给定一个先验分布 $P_{prior}(z)$ 并使其通过生成器 G ，于是就可以得到一个新的概率分布 $P_{G}(x)$，并使 $P_{G}(x)$ 与 $P_{data}(x)$ 越来越接近。
- 然后，定义一个判别器 D，它的输入是 x，输出是一个标量，它能够通过某种方法计算 $P_{G}(x)$ 和 $P_{data}(x)$ 之间的相似性。

对于整个 GAN 来说，上面两个步骤是通过下式来实现的：

$$G^* = arg \min _G \max _D V(G,D)=arg\min _G \max _D \mathbb{E} _{x \sim P_{data}} [\log D(x)] + \mathbb{E} _{x \sim P_{G}} [\log (1-D(x))] \tag{4}$$

　　接下来将对公式(4)数学意义进行解释说明。首先思考一个问题，在给定生成器 G 的条件下，什么样的 $D^*$ 能够使 V(G,D) 最大？首先对 $V(G,D)$ 进行一定的变换可得到：

$$
\begin{align*}
V(G,D) & = \mathbb{E} _{x \sim P_{data}} [\log D(x)] + \mathbb{E} _{x \sim P_{G}} [\log (1-D(x))] \\
& = \int _{x} P_{data}(x) \log D(x) dx + \int _x P_{G}(x) \log (1-D(x)) dx \\
& = \int _x [P_{data}(x) \log D(x) + P_{G}(x) \log (1-D(x))] dx \tag{5}
\end{align*}
$$

　　要想使上式最大，其实就是要使得中括号内的式子值最大，因此我们对其求偏导可得：
$$
\frac{\partial [P_{data}(x) \log D(x) + P_{G}(x) \log (1-D(x)]}{\partial D(x)} = \frac{P_{data}(x)}{D(x)} - \frac{P_{G}(x)}{1-D(x)} \tag{6}
$$


　　然后令上式为零，可以解得 $D^*(x) = \frac{P_{data}(x)}{P_{data}(x) + P_G(x)}$ ，再将 $D^{\*}(x)$ 带入到 $V(G,D)$ 就可以得到：

$$
\begin{align*}
\max _D V(G,D) &= V(G, D^*) \\
& = \mathbb{E} _{x \sim P_{data}} [\log \frac{P_{data}(x)}{P_{data}(x) + P_{G}(x)}] + \mathbb{E} _{x \sim P_{G}} [\log \frac{P_{G}(x)}{P_{data}(x) + P_{G}(x)}] \\
& = \int _x P_{data}(x) \log \frac{P_{data}(x)}{P_{data}(x) + P_{G}(x)} dx + \int _x P_{G}(x) \log \frac{P_{G}(x)}{P_{data}(x) + P_{G}(x)} dx \\
& = \int _x P_{data}(x) \log \frac{ \frac{1}{2}P_{data}(x)}{\frac{P_{data}(x) + P_{G}(x)}{2}} dx + \int _x P_{G}(x) \log \frac{\frac{1}{2}P_{G}(x)}{\frac{P_{data}(x) + P_{G}(x)}{2}} dx \\
& = -2log2 + KL(P_{data}(x) || \frac{P_{data}(x) + P_{G}(x)}{2}) + KL(P_{G}(x) || \frac{P_{data}(x) + P_{G}(x)}{2})\\
& = -2log2 + 2JSD(P_{data}(x) || P_G(x)) \tag{7}
\end{align*}
$$

　　从公式(7)我们可以很直观的看到， $\max_{D} V(G,D)$  实际上是等价于求两个分布 $P_G(x)$  和  $P_{data}(x)$ 之间的 JS 散度，而 JS 散度也是衡量两个分布相似性的大小。

最后，寻找一个 $G^*$ 满足：

$$G^* = arg \min _G \max _D V(G,D)=arg \min _G [-2log2 + 2JSD(P_{data}(x) || P_G(x))] \tag{8}$$

根据 JS 散度的性质可以知道，当且仅当 $P_G(x) = P_{data}(x)$ 时上式最小。

　　到这里我们就明白了，通过前面那么一堆的推导过程，我们发现公式(4)的目的和最大似然估计一样，都是希望通过对 $P_{G}(x; \theta)$ 中参数 $\theta$ 进行调整，使得 $P_G(x) = P_{data}(x)$。 这与我们目标是一致，希望寻找一个分布能够尽可能地拟合真实的样本分布。所以 GAN 通过该方法能够使得在无法计算似然函数的时候也能够求解一个分布让其尽可能地拟合真实数据的分布。

　　接下来就只剩下一个问题了，就是如何对公式(4)进行求解了。之前已经提到过，GAN 的训练过程可以看成是一个演化过程，之前可能不太理解这种训练的方式的缘由，但通过上面的公式推导之后，我们对这个演化的过程就应该有了一个更加深刻的认识了：

- 给定一个 $G_0$
- 找到一个 $D_0$   使得   $V(G_0,D)$ 的值最大， 而根据之前的推导知道 $V(G_0,D_0^*)$ 实际上测量的就是 $P_{data}(x)$ 和 $P_{G_0}(x)$ 之间的 JS 散度。
- 再通过梯度下降法，更新生成器的参数 $\theta$，得到新的生成器 $G_1$，使得 $P_{data}(x)$ 和 $P_{G_0}(x)$ 之间的散度变小，即： $P_{data}(x)$ 和 $P_{G_0}(x)$ 更相似。
- 再找到一个 $D_{1}$   使得 $V(G_1, D)$ 最大，$V(G_1, D_1^*)$ 实际上测量的就是 $P_{data}(x)$ 和 $P_{G_1}(x)$ 之间的 JS 散度。
- ......重复2、3步，使得其最终生成比较好的结果。



#### **GAN 的实际操作**

　　上述都是对 GAN 的理论进行介绍，接下来简单介绍一下 GAN 在实际中如何进行训练。

　　在给定一个 G 的情况下，实际中如何计算 $\max_DV(G,D)$ ？由于 $P_{data}(x)$ 和 $P_{G_0}(x)$ 期望我们无法计算，因此我们采用的方法是采样，分别从 $P_{data}(x)$ 和 $P_G(x)$ 采样 m 个样本 ${x^1,x^2,...,x^m}$ 和 ${\widetilde{x}^1, \widetilde{x}^2, ... , \widetilde{x}^m}$ 。因此原来的

$$\max _D V(G,D)=\mathbb{E} _{x \sim P_{data}} [\log D(x)] + \mathbb{E} _{x \sim P_{G}} [\log (1-D(x))] \tag{9}$$

　　就变成了

$$\max _D \widetilde{V}(G,D)=\frac{1}{m} \sum_{i=1}^{m} [\log D(x^i)] + \frac{1}{m} \sum_{i=1}^{m}  [\log (1-D(\widetilde{x}^i))] \tag{10}$$

　　然后在给定 D 的情况下，训练的 G 的式子也就变成了：

$$\min _G\max _D \widetilde{V}(G,D)=\frac{1}{m} \sum_{i=1}^{m} [\log D(x^i)] + \frac{1}{m} \sum_{i=1}^{m}  [\log (1-D(G(z^i)))] \tag{11}$$


### 小结
　　到此为止，GAN 的相关介绍就结束了，这篇博文主要是对 GAN 中的数学原理进行了一个简单推导，帮助那些初学者对 GAN 有一个更加深刻和系统的理解。后续博文将会对 f-divergence GAN 以及 WGAN 等 GAN 的变型进行一些介绍，它们的基本原理都是在这些基础上进行一些相关改变的，因此弄懂这些后，对它们的理解将会变得容易一些。

<div align="center">
	<img src="/images/posts/GAN/ppgn.png" height="300" width="600">
</div>
