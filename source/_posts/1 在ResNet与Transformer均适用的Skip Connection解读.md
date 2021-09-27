---
title:  在ResNet与Transformer均适用的Skip Connection解读
categories:
  - 卷积CNN
comments: true
copyright_author: ChaucerG
date: 2021-09-05 17:16:48
tags:
- 残差连接
- CNN
- Tansformer
- ResNet
keywords:
- 残差连接
- CNN
- Tansformer
- ResNet
description:
- 该文主要是分析和讨论了跳跃连接的一些局限，同时分析了BN的一些限制，提出了通过递归的Skip connection和layer normalization来自适应地调整输入scale的策略，可以很好的提升跳Skip connection的性能，该方法在CV和NLP领域均适用。
top_img:
cover:
---

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/1.png)

> 该文主要是分析和讨论了跳跃连接的一些局限，同时分析了BN的一些限制，提出了通过递归的Skip connection和layer normalization来自适应地调整输入scale的策略，可以很好的提升跳Skip connection的性能，该方法在CV和NLP领域均适用。

## 简介
Skip connection是一种广泛应用于提高深度神经网络性能和收敛性的技术，它通过神经网络层传播的线性分量，缓解了非线性带来的优化困难。但是，从另一个角度来看，它也可以看作是输入和输出之间的调制机制，输入按预定义值1进行缩放。

在本文中，作者通过研究Skip connection的有效性和scale factors显示，一个微不足道的调整将导致spurious gradient爆炸或消失，这可以通过normalization来解决，特别是layer normalization。受此启发作者进一步提出通过递归的Skip connection和layer normalization来自适应地调整输入scale，这大大提高了性能，并且在包括机器翻译和图像分类数据集在内的各种任务中具有很好的泛化效果。

![图1 常用skip connections](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/2.png)

### 这项工作的特点：
1) 主要关注LN和skip connection的结合；
2) 重新思考了层归一化的作用，选择不进行缩放；
3) 在具有代表性的计算机视觉和自然语言处理任务上进行实验；
4) 摆脱了泛化了所有以前工作的残差块的一般形式，并提出了一种新的递归残差块结构，它具有层归一化，优于本工作中检查的所有一般形式的变体；

## 方法

### connection problem
在进行尺度scaling时，会出现梯度爆炸或消失的问题，阻碍了深度神经网络的高效优化。

### optimization problem
由于早期的工作已经确定，将Skip connection直接结合到神经网络的前向传播中就足够了，不需要任何尺度，后续的优化问题研究大多遵循Skip connection结构。

### 架构说明

![图2 常见LN与skip connections组合](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/3.png)

#### **Expanded Skip Connection (xSkip)**：

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/4.png)

其中，$x$和$y$分别为残差块的输入和输出。$F$为weighted neural network layer，$\lambda$为modulating scalar。

考虑到神经网络层可能具有不同的表示能力和优化难度，这种结构自然调整了跳跃的重要性。然而，需要注意的是，在这项工作中$\lambda$是固定的，目的是隔离缩放的影响。虽然学习过的$\lambda$可能更好地捕捉到这2个部分之间的平衡，但是学习$\lambda$变成了另一个变量。

#### **Expanded Skip Connection with Layer Normalization (xSkip+LN)**：
在Transformer将跳跃连接与层规范化相结合的激励下，作者进一步研究了层规范化对扩展跳跃连接的影响：

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/5.png)

实验表明层归一化有助于缓解调制因子在优化过程中引起的梯度畸变。不同于作用于“样本空间”的BN，LN则是作用于“特征空间”。同时在神经网络难以优化的情况下，LN仍然可以帮助学习shortcut，而BN可能会失败。

#### **Recursive Skip Connection with Layer Normalization (rSkip+LN)**：
另一种稳定梯度的方法是每次保持$\lambda$=1，但重复添加带有LN的shortcut，这样更多的输入信息也被建模。它被递归定义为：

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/6.png)

$\lambda$应该是一个不小于1的整数。例如，当$\lambda$=1时，上式便回归到Transformer中使用的block，并符合跳过不需要缩放的结果。

通过recursive skip connection with layer normalization，该模型鼓励多次使用层归一化来改进优化，通过跳跃连接可以包含更多的x信息。此外，与一次性简单地合并比例跳跃相比，该模型可能获得更强的表达能力，因为每一个递归步骤本质上构建了一个不同的特征分布，递归结构可以学习自适应的x与F(x,W)。

## 实验
### 实验1：PreAct-ResNet-110 on cifar10
![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/7.png)

### 实验2：EN-VI machine translation

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/8.png)

### 实验3：BN代替LN

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210521/9.png)

可以看出，与LN结合跳跃连接相比，BN的效果较差。而本文所提出的递归策略可以帮助BN提升效果。

### 实验结论
作者通过对不同任务的实验（Transformer和ResNet），得出如下结论:

- 没有经过任何归一化的expanded skip connection确实会造成梯度畸形，导致神经网络的学习效果不理想。层归一化在一定程度上有助于解决 expanded skip connection带来的优化问题。

- 本文提出的带有LN的recursive skip connection，通过将expanded skip connection划分为多个阶段，以更好地融合转换输入的效果，进一步简化了优化过程。

- 利用Transformer在WMT-2014 EN-DE机器翻译数据集上的实验结果进一步证明了递归架构的有效性和效率，模型性能甚至优于3倍大的模型。

## 参考
[1].Rethinking Skip Connection with Layer Normalization in Transformers and ResNets<br>

