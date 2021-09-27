---
title: 详细解读Transformer那些有趣的特性
categories:
  - Transformer
comments: true
copyright_author: ChaucerG
date: 2021-09-05 19:35:45
tags:
 - Transformer
keywords:
 - Transformer
description:
- 本文发现了Transformer的一些重要特性，如Transformer对严重的遮挡，扰动和域偏移具有很高的鲁棒性、与CNN相比，ViT更符合人类视觉系统，泛化性更强，等等...  代码即将开源！
top_img:
cover:
---

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/1.png)

>本文发现了Transformer的一些重要特性，如**Transformer对严重的遮挡，扰动和域偏移具有很高的鲁棒性**、**与CNN相比，ViT更符合人类视觉系统，泛化性更强**，等等...  代码即将开源！<br>
**作者单位**：澳大利亚国立大学, 蒙纳士大学, 谷歌等7家高校/企业
## 简介

近期Vision Transformer（ViT）在各个垂直任务上均表现出非常不错的性能。这些模型基于multi-head自注意力机制，该机制可以灵活地处理一系列图像patches以对上下文cues进行编码。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/2.png)


![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/3.png)


一个重要的问题是，在以给定patch为条件的图像范围内，如何灵活地处理图像中的干扰，例如严重的遮挡问题、域偏移问题、空间排列问题、对抗性和自然扰动等等问题。作者通过涵盖3个ViT系列的大量实验，以及与高性能卷积神经网络（CNN）的比较，系统地研究了这些问题。并通过分析得出了ViT的以下的特性：

1) Transformer对严重的遮挡，扰动和域偏移具有很高的鲁棒性，例如，即使随机遮挡80％的图像内容，其在ImageNet上仍可保持高达60％的top-1精度; 

2) Transformer对于遮挡的良好表现并不是由于依赖局部纹理信息，与CNN相比，ViT对纹理的依赖要小得多。当经过适当训练以对基于shape的特征进行编码时，ViT可以展现出与人类视觉系统相当的shape识别能力;

3) 使用ViT对shape进行编码会产生有趣的现象，在即使没有像素级监督的情况下也可以进行精确的语义分割;

4) 可以将单个ViT模型提取的特征进行组合以创建特征集合，从而在传统学习模型和少量学习模型中的一系列分类数据集上实现较高的准确率。实验表明，ViT的有效特征是由于通过自注意力机制可以产生的灵活和动态的感受野所带来的。

## 本文讨论主题
### 2.1 ViT对遮挡鲁棒否？
这里假设有一个网络模型$f$，它通过处理一个输入图像$x$来预测一个标签$y$，其中$x$可以表示为一个patch $x=\{x_i\}_{i=1}^N$的序列，$N$是图像patch的总数。

虽然可以有很多种方法来建模遮挡，但本文还是选择了采用一个简单的掩蔽策略，选择整个图像patch的一个子集，$M < N$，并将这些patch的像素值设为0，这样便创建一个遮挡图像$x'$。

作者将上述方法称为**PatchDrop**。目的是观察鲁棒性$f(x')_{argmax}=y$。

作者总共实验了3种遮挡方法:
1) Random PatchDrop
2) Salient(foreground) PatchDrop
3) Non-salient (background) PatchDrop

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/4.png)

#### ***1、Random PatchDrop***
ViT通常将图像划分为196个patch，每个patch为14x14网格，这样一幅224x224x3大小的图像分割成196个patches，每个patch的大小为16x16x3。例如，随机从输入中删除100个这样的补丁就相当于丢失了51%的图像内容。而这个随机删除的过程即为**Random PatchDrop**。

#### ***2、Salient(foreground) PatchDrop***
对于分类器来说，并不是所有的像素都具有相同的值。为了估计显著区域，作者利用了一个自监督的ViT模型DINO，该模型使用注意力分割图像中的显著目标。按照这种方法可以从196个包含前n个百分比的前景信息的patches中选择一个子集并删除它们。而这种通过自监督模型删除显著区域的过程即为**Salient (foreground) PatchDrop**。

#### ***3、Non-salient(background) PatchDrop***
采用与SP（Salient(foreground) PatchDrop）相同的方法选择图像中最不显著的区域。包含前景信息中最低n%的patch被选中并删除。同样，而这种通过自监督模型删除非显著区域的过程即为**Non-salient(background) PatchDrop**。


#### 鲁棒性分析

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/5.png)

以Random PatchDrop为例，作者给出5次测试的平均值和标准偏差。对于显著性和非显著性Patchdrop，由于获得的遮挡掩模是确定性的，作者只给出了1次运行的精度值。

Random PatchDrop 50%的图像信息几乎完全破坏了CNN的识别能力。例如，当去掉50%的图像内容时ResNet50的准确率为0.1%，而DeiT-S的准确率为70%。一个极端的例子可以观察到，当90%的图像信息丢失，但Deit-B仍然显示出37%的识别精度。这个结果在不同的ViT体系结构中是一致的。同样，ViT对前景(显著)和背景(非显著)内容的去除也有很不错的表现。

#### Class Token Preserves Information
为了更好地理解模型在这种遮挡下的性能鲁棒的原有，作者将不同层的注意力可视化(图4)。 通过下图可以看出浅层更关注遮挡区域，而较深的层更关注图像中的遮挡以外的信息。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/6.png)

然后作者还研究这种从浅层到更深层次的变化是否是导致针对遮挡的Token不变性的原因，而这对分类是非常重要的。作者测量了原始图像和遮挡图像的特征/标记之间的相关系数。在ResNet50的情况下测试在logit层之前的特性，对于ViT模型，Class Token从最后一个Transformer block中提取。与ResNet50特性相比，来自Transformer的Class Token明显更鲁棒，也不会遭受太多信息损失(表1)。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/7.png)

此外，作者还可视化了ImageNet中12个选择的超类的相关系数，并注意到这种趋势在不同的类类型中都存在，即使是相对较小的对象类型，如昆虫，食物和鸟类。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/8.png)


### 2.2 ViT能否同时学习Shape和Texture这2种特性？
Geirhos等人引入了Shape vs Texture的假设，并提出了一种训练框架来增强卷积神经网络(CNNs)中的shape偏差。

首先，作者对ViT模型进行了类似的分析，得出了比CNN更强的shape偏差，与人类视觉系统识别形状的能力相当。然而，这种方法会导致自然图像的精度显著下降。

为了解决这种问题，在第2种方法中，作者将shape token引入到Transformer体系结构中，专门学习shape信息，使用一组不同的Token在同一体系结构中建模Shape和Texture相关的特征。为此，作者从预训练的高shape偏差CNN模型中提取shape信息。而作者的这种蒸馏方法提供了一种平衡，既保持合理的分类精度，又提供比原始ViT模型更好的shape偏差。

#### Training without Local Texture
在训练中首先通过创建一个SIN风格化的ImageNet数据（从训练数据中删除局部纹理信息）。在这个数据集上训练非常小的DeiT模型。通常情况下，vit在训练期间需要大量的数据增强。然而，由于较少的纹理细节，使用SIN进行学习是一项困难的任务，并且在风格化样本上进行进一步的扩展会破坏shape信息，使训练不稳定。因此，在SIN上训练模型不使用任何augmentation、label smoothing或Mixup。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/9.png)

作者观察到，在ImageNet上训练的ViT模型比类似参数量的CNN模型表现出更高的shape偏差，例如，具有2200万个参数的DeiT-S比ResNet50表现更好(右图)。当比较SIN训练模型时，ViT模型始终优于cnn模型。有趣的是，DeiT-S在SIN数据集上训练时达到了人类水平(左图)。

#### Shape Distillation
通过学习Teacher models 提供的soft labels，知识蒸馏可以将大teacher models压缩成较小的Student Model。本文作者引入了一种新的shape token，并采用 Adapt Attentive Distillation从SIN dataset(ResNet50-SIN)训练的CNN中提取Shape特征。作者注意到，ViT特性本质上是动态的，可以通过Auxiliary Token来控制其学习所需的特征。这意味着单个ViT模型可以同时使用单独的标记显示high shape和texture bias(下表)。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/10.png)

当引入shape token时在分类和形状偏差度量方面获得了更平衡的性能(图6)。为了证明这些不同的token(用于分类和shape)可以确实模型独特的特征，作者计算了所蒸馏的模型DeiT-T-SIN和DeiT-S-SIN的class和shape token之间的余弦相似度，结果分别是0.35和0.68。这明显低于class和distill token之间的相似性；DeiT-T和Deit-S分别为0.96和0.94。这证实了关于在Transformer中使用单独的Token可以用来建模不同特征的假设，这是一种独特的能力，但是不能直接用在CNN模型中。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/11.png)

#### Shape-biased ViT Offers Automated Object Segmentation

有趣的是，没有局部纹理或形状蒸馏的训练可以让ViT专注于场景中的前景物体而忽略背景(图4)。这为图像提供了自动语义分割的特征，尽管该模型从未显示像素级对象标签。这也表明，在ViT中促进shape偏差作为一个自监督信号，模型可以学习不同shape相关的特征，这有助于定位正确的前景对象。值得注意的是，没有使用shape token的训练中ViT表现得比较差(Table 3)。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/12.png)

### 2.3 位置编码是否真的可以表征Global Context？
Transformer使用self-attention(而不是RNN中的顺序设计)并行处理长序列，其序列排序是不变的。但是它的明显缺点是忽略了输入序列元素的顺序，这可能很重要。

在视觉领域patch的排列顺序代表了图像的整体结构和整体构成。由于ViT对图像块进行序列处理，改变序列的顺序，例如对图像块进行shuffle操作但是该操作会破坏图像结构。

当前的ViT使用位置编码来保存Context。在这里问题是，如果序列顺序建模的位置编码允许ViT在遮挡处理是否依然有效?

然而，分析表明，Transformer显示排列不变的patch位置。位置编码对向ViT模型注入图像结构信息的作用是有限的。这一观察结果也与语言领域的发现相一致。

#### Sensitivity to Spatial Structure
通过对输入图像patch使用shuffle操作来消除下图所示的图像(空间关系)中的结构信息。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/13.png)

作者观察到，当输入图像的空间结构受到干扰时，DeiT模型比CNN模型保持了更高程度的准确性。这也一方面证明了位置编码对于做出正确的分类决策并不是至关重要的，并且该模型并没有使用位置编码中保存的patch序列信息来恢复全局图像context。即使在没有这种编码的情况下，与使用位置编码的ViT相比，ViT也能够保持其性能，并表现出更好的排列不变性(下图)。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/14.png)

最后，在ViT训练过程中，当patch大小发生变化时，对自然图像进行非混叠处理时，其排列不变性也会随着精度的降低而降低(下图)。作者将ViT的排列不变性归因于它们的动态感受野，该感受野依赖于输入小patch，可以与其他序列元素调整注意，从而在中等变换速率下，改变小patch的顺序不会显著降低表现。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/15.png)

>从上面的分析可以看出，就像texture bias假设是错误的一样，依赖位置编码来在遮挡下表现良好也是不准确的。作者得出这样的结论，这种鲁棒性可能只是由于ViT灵活和动态的感受野所带来的，这同时也取决于输入图像的内容。

### 2.4 ViT对对抗信息和自然扰动的鲁棒性又如何？
作者通过计算针对雨、雾、雪和噪声等多种综合常见干扰的平均损坏误差(mCE)来研究这一问题。具有类似CNN参数的ViT(例如，DeiT-S)比经过增强训练的ResNet50(Augmix)对图像干扰更加鲁棒。有趣的是，在ImageNet或SIN上未经增强训练的卷积和Transformer模型更容易受到图像干扰的影响(表6)。这些发现与此一致，表明数据增强对于提高常见干扰的鲁棒性是很必要的。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/16.png)

作者还观察到adversarial patch攻击的类似问题。ViT的鲁棒性高于CNN，通用adversarial patch在白盒设置(完全了解模型参数)。在SIN上训练的ViT和CNN比在ImageNet上训练的模型更容易受到adversarial patch攻击(图10)，这是由于shape偏差与鲁棒性权衡导致的。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/17.png)

### 2.5 当前ViT的最佳Token是什么？
ViT模型的一个独特特征是模型中的每个patch产生一个class token，class head可以单独处理该class token(下图所示)。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/18.png)

使得可以测量一个ImageNet预先训练的ViT的每个单独patch的区分能力，如图12所示，由更深的区块产生的class token更具鉴别性，作者利用这一结果来识别其token具有best downstream transferability最优patch token集合。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/19.png)

#### Transfer Methodology
如图12所示，作者分析了DeiT模型的block的分类精度，发现在最后几个block的class token中捕获了最优的判别信息。为了验证是否可以将这些信息组合起来以获得更好的性能，作者使用DeiT-S对细粒度分类数据集上现成的迁移学习进行了消融研究(CUB)，如下表所示。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/20.png)

在这里，作者从不同的块连接class token(可选地结合平均补丁标记)，并训练一个线性分类器来将特征转移到下游任务。

请注意，一个patch token是通过沿着patch维度进行平均生成的。最后4个块的class token连接得到了最好的迁移学习性能。

作者将这种迁移方法称为**DeiT-S(ensemble)**。来自所有块的class token和averaged patch tokens的拼接表现出与来自最后4个块的token相似的性能，但是需要显著的大参数来训练。作者进一步在更大范围的任务中使用DeiT-S(集成)进行进一步实验，以验证假设。在接下来的实验中，同时还将CNN Baseline与在预训练的ResNet50的logit层之前提取的特征进行比较。

#### General Classification
作者还研究了几个数据集的现成特征的可迁移性，包括Aircraft, CUB, DTD, GTSRB, Fungi, Places365和iNaturalist数据集。这些数据集分别用于细粒度识别、纹理分类、交通标志识别、真菌种类分类和场景识别，分别有100、200、47、43、1394、365和1010类。在每个数据集上训练一个线性分类器，并在测试分割上评估其性能。与CNN Baseline相比，ViT特征有了明显的改善(图13)。事实上，参数比ResNet50少5倍左右的DeiT-T性能更好。此外，本文提出的集成策略在所有数据集上都获得了最好的结果。


#### Few-Shot Learning
在few-shot learning的情况下，元数据集是一个大规模的benchmark，包含一个不同的数据集集覆盖多个领域。作者使用提取的特征为每个query学习support set上的线性分类器，并使用标准FSL协议评估。ViT特征在这些不同的领域之间转移得更好(图13)。作者还强调了QuickDraw的一个改进，包含手绘草图的数据集，这与研究结果一致。

![](https://gitee.com/chaucerg/pic_-web/raw/master/images_20210525/21.png)

## 参考
[1].Intriguing Properties of Vision Transformers.<br>
