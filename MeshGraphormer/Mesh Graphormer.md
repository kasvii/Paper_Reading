# Mesh Graphormer

## 摘要

本文提出了一个**图卷积增强的transformer**，名为Mesh Graphormer，用来从单张图片进行三维人体姿态和网格重建。最近，**transformer和GCNN图卷积神经网络**在人体三维重建产生了极大（promising）的进步。基于transformer的方法能够有效建模三维网格顶点和关节的非局部交互，GCNNs擅长利用基于预先指定的网格拓扑探索相邻顶点的交互。本文研究了如何将图卷积和自注意力结合的transformer来建模**局部和全局的交互**。实验结果显示Mesh Graphormer在多个基准（benchmarks），Human3.6M、3DPW和FreiHAND，取得了出色的结果。

## 引用

从单幅图像中进行人体三维重建是一个热门的研究课题，因为它为人机交互提供了广泛的应用。然而，由于身体关节的复杂性，让它成为一项具有挑战性的任务。

最近，Transformer和GCNNs在促进了人体三维重建的进步。[1]Pose2mesh和[2]使用GCNNs利用相邻顶点之间的局部交互作用，直接回归顶点的三维位置。在[3]METRO中使用了transformer的编码器，用自注意力来获取关节点和网格顶点的全局交互，有效提高了性能。

但是，transformers和CNN都有他们的局限，**transformer擅长建模长距离的依赖，但是在捕获细粒度的局部信息方面效率低下**。另一方面呢，**卷积层对提取局部特征很有用，但是需要许多层来捕获全局上下文**。在NLP上最近提出了Conformer[4]，它利用**自注意力**和**卷积**的互补性来学习表示。启发我们结合自注意力和图卷积进行人体三维重建。

我们提出了一个**图卷积增强的transformer**，名为Mesh Graphormer，用来从单张图片进行三维人体姿态和网格重建。将图卷积注入到transformer中，来改善相邻顶点和关节点的交互。为了充分利用图卷积的能力，**Graphormer**可以自由处理所有的包含详细局部信息的**图像网格特征**，有助于细化三维坐标的预测。所以Graphormer和图像网格特征能够相互增强来实现更好的性能。

主要的贡献有：

- 我们提出了一种图卷积增强的Transformer，名为Mesh Graphormer，以建模局部和全局交互的三维重建的人体姿态和网格。
- Mesh Graphormer允许**关节**和**网格顶点**自由地关注**图像网格特征**，来细化三维坐标的预测。
- Mesh Graphormer在Human3.6M, 3DPW和FreiHAND数据集上表现超过SOTA

## 相关工作

### 人体三维重建

人体三维重建可以分为有模型或者无模型方法，大量的工作使用SMPL之类的人体模型，并以SMPL参数空间作为回归目标。但是，从单一图像中估计准确的系数还是很困难，研究TCMR、VIBE、pose2pose[5678]正试图预测三维姿态，学习使用更多的视觉特征或者采用稠密的关联映射来改善重建。

除了回归模型参数，非参数的方法，比如[9 10 11]Pose2mesh、I2L-MeshNet**直接从图像中回归顶点**。在这些方法中GCNN是最受欢迎的选择之一，因为他能够基于给定的邻接矩阵建模相邻顶点的局部交互。但GCNN捕获顶点和关节之间的全局交互的效率低下。为了克服这一限制，基于transformer的[3]METRO使用自注意力机制来自由地关注顶点和身体关节，从而编码人体网格的非局部关系。然而，建立局部交互模型不如基于GNN的方法方便。

本文的方法和METRO的不同主要在于：

1. 我们设计了一个图卷积增强Transformer编码器。
2. 我们加入图像网格特征作为输入，加入到transformer，并允许关节点和网格顶点关注网格特征。

### Trasnformer 架构

Tranformer发展的一个重要方向是提高Transformer网络的表达性，以便更好地进行上下文建模。最近的工作显示结合卷积和自注意力到transformer编码器能够提高表示学习（representation learning）。

为了解决这些挑战，我们研究了如何**将图卷积注入到transformer，以更好地建模三维网格顶点和人体关节之间的局部和全局交互**。

## Graphormer编码器

框架如图：

![1](MeshGraphormer/1.png)

我们的Graphormer和传统的Transformer有相似的结构，但我们引入了图卷积来建模细粒度的局部交互。

### 多头自注意力

多头自注意力并行使用许多自注意力函数来学习上下文表示。

![2](MeshGraphormer/2.png)

![3](MeshGraphormer/3.png)

### 图残差块

图残差块用来铺货细粒度的局部信息。

将MHSA产生的上下文特征Y通过图卷积提高局部交互：

![4](MeshGraphormer/4.png)

采用的激活函数是GeLU（参考BERT）

图卷积块的搭建以GraphCMR[12]为模板，替换：

- group normalization ->layer nomalization
- ReLU -> GeLU

我们的图残差块显式编码图结构，从而提高了特征中的空间局部性。

## 网格重建中的Graphormer

![5](MeshGraphormer/5.png)

图3(a)给出了端到端网格回归框架。以大小为224x224的图像作为输入，提取**图像网格特征**。这些图像特征作为一个多层Graphormer编码器的输入。最后，我们的端到端框架同时预测了网格顶点和身体关节的三维坐标。

### CNN和图像网格特征

之前的工作提取一个全局2048为的图像特征向量作为模型输入，不足在于全局特征向量并不包含细粒度的局部特征。这促使我们加入网格特征作为输入标记，并允许关节和网格顶点自由地关注所有的网格特征（grid features）。网格特征提供的局部信息可以有效地利用Graphormer的图卷积来细化网格顶点和身体关节的三维信息。

如图3(a)所示，我们从CNN的最后一个卷积块中提取网格特征。网格特征是典型的7×7×1024大小。他们被标记为49个token，每个token是1024维向量。我们也从CNN最后一层隐藏层提取了2048维的图像特征向量，并利用每个顶点和人体关节的三维坐标进行位置编码。最后我们用MLP统一所有输入的大小。最后，所有输入token有2051维。

### 多层Graphormer编码器

给定网格特征、关节查询和顶点查询，我们的多层Graphormer编码器依次减少维度，将输入同时映射到三维人体关节和网格顶点映射。如图3(b)所示。有三个编码块，每个编码块有相同的token包括49个网格特征token、14个关节查询和431个顶点查询。

我们的多层Graphormer编码器用431个顶点，用于位置编码，产生一个粗糙的mesh，然后我们使用线性投影对粗mehs上采样达到原始分辨率（SMPL拓扑结构中有6k个顶点）。学习粗网格后进行上采样有助于避免原始网格中的冗余，使训练更有效。

### 训练细节

和METRO一样，我们在Graphormer输出中加入了L1损失。在训练中，还加入了METRO的Masked Vertex Modeling来提高模型鲁棒性。特别的，我们将L1损失加到三维顶点和关节中，还加入到2维关节投影来改善图像和重建网格的对齐。另外，我们对粗mesh进行中间监督来加速收敛。

优化器：Adam，学习率：$10^{-4}$（Graphormer和CNN），epoch：200，每100个epoch学习率降低10倍；Graphormer权重都是随机初始化，CNN用ImageNet预训练权重初始化。

## 实验

能够进行人体和人手重建

### 主要结果

![6](MeshGraphormer/6.png)

图四显示我们的模型对**挑战性的姿势和噪声**背景更具鲁棒性。

![7](MeshGraphormer/7.png)

### 消融实验

- **网格特征的有效性**：

  ![8](MeshGraphormer/8.png)

  本文除了常用的图像全局特征，还加入了**图像网格特征**（局部信息），在消融实验中验证了图像网格特征对重建性能的提升。说明只靠单一的全局特征向量是现有技术中的性能瓶颈之一。

  <img src="MeshGraphormer/12.png" alt="12" style="zoom:90%;" /><img src="MeshGraphormer/13.png" alt="13" style="zoom:90%;" />

- **图卷积的有效性**：

  ![9](MeshGraphormer/9.png)

  结果显示了一些有趣的情况：
  
  - 在编码器1或编码器2中添加一个图卷积并不能提高性能。
  
  - 将图卷积添加到编码器3中，提高了0.9PA-MPJPE。
  
    结果表明，**低层更关注网格顶点的全局交互**作用来模拟人体的姿态，而**高层更关注局部交互作用**，以实现更好的形状重建。

- **分析编码器结构**：

  ![10](MeshGraphormer/10.png)

  探究MHSA和图卷积模块的关系：

  - 图卷积层和MHSA并行（表6，第一行）
  - 先图卷积，再MHSA（第二行）
  - 先MHSA，再图卷积（第三行，性能最好）

- **加入图残差块的有效性**：

  ![11](MeshGraphormer/11.png)

  将图卷积层替换成图残差块

  表7展示图残差块比图卷积层性能更好

- **网格特征与图卷积的关系**：

  ![14](MeshGraphormer/14.png)

  当我们同时使用网格特征和图卷积时，它最终将PA-MPJPE提高了2.2，这远远大于两个单独改进的总和（0.1+0.8）。结果表明，网格特征和图的卷积相互强化。

- **局部交互的可视化**：

  ![15](MeshGraphormer/15.png)

  从编码器3（最后一层）提取注意力图，并计算了所有attention head的平均注意力。在图7中，我们看到METRO未能建模左膝和左脚趾之间的相互作用。相比之下，Graphormer能够建模全局和局部的交互，特别是左膝和左脚趾，建模效果更好。

## 总结

本文提出了**Mesh Graphormer**，一种结合**图卷积**和**自注意力**的**Transformer**架构，用来从**单张图片**重建人体姿态和网格模型。我们探索了多种设计方案，并证明了图卷积和**网格特征**有助于提高Transformer的性能。

## INTUITIAN

- 本文除了常用的图像全局特征，还加入了**图像网格特征**（局部信息），在消融实验中验证了图像网格特征对重建性能的提升。说明只靠单一的全局特征向量是现有技术中的性能瓶颈之一。
- **特征金字塔**有助于提高Transformer编码器的性能。
- Graphormer中，**低层更关注网格顶点的全局交互**作用来模拟人体的姿态，而**高层更关注局部交互作用**，

## 数据集

**Human3.6M**: Catalin Ionescu, Dragos Papava, Vlad Olaru, and Cristian Sminchisescu. Human3.6M: Large scale datasets and predictive methods for 3D human sensing in natural environments. In *IEEE Transaction on Pattern Analysis and Machine Intelligence*, 2014. 

**3DPW**: Timo von Marcard, Roberto Henschel, Michael Black, Bodo Rosenhahn, and Gerard Pons-Moll. Recovering accurate 3D human pose in the wild using IMUs and a moving camera. In *European Conference on Computer Vision*, 2018.

**MPI-INF-3DHP**: Dushyant Mehta, Helge Rhodin, Dan Casas, Pascal Fua, Oleksandr Sotnychenko, Weipeng Xu, and Christian Theobalt. Monocular 3D human pose estimation in the wild using improved CNN supervision. In *International Conference on 3DVision*, 2017.

**UP-3D**:  Unite the people: Closing the loop between 3d and 2d human representations. In *CVPR*, 2017.

**MuCo-3DHP**:  Single-shot multi-person 3d pose estimation from monocular rgb. In *3DV*, 2018.

**COCO**: Microsoft coco: Common objects in context. In *ECCV*, 2014.

**MPII**: 2d human pose estimation: New benchmark and state of the art analysis. In *CVPR*, 2014.

**FreiHAND**: Freihand: A dataset for markerless capture of hand pose and shape from single rgb images. In *ICCV*, 2019.

## 专业词汇

GCNN: Graph Convolutional Neural Network - 图卷积神经网络

self-attention - 自注意力

visual evidence - 视觉特征

correspondence maps - 关联映射？？？

adjacency matrix - 邻接矩阵

expressiveness - 表达

representation learning - 表示学习：学习特征

Multi-head Self Attention（MHSA）- 多头注意力

linear projection - 线性投影

mask vertex modeling - 掩膜顶点建模

pseudo data - 伪数据

feature map - 特征图

attention head - 注意力头？？？



## 写作词汇

benchmarks - 基准

promising - 有希望的，有前途的

complementarity - 互补性

fine-grained - 细粒度的

convergence - 收敛



## 参考文献

[1] Hongsuk Choi, Gyeongsik Moon, and Kyoung Mu Lee.

Pose2mesh: Graph convolutional network for 3d human pose

and mesh recovery from a 2d human pose. In *ECCV*, 2020.

[2] Nikos Kolotouros, Georgios Pavlakos, and Kostas Dani

ilidis. Convolutional mesh regression for single-image hu

man shape reconstruction. In *CVPR*, 2019.

[3] METRO: End-to-end human pose and mesh reconstruction with transformers. CVPR 2021.

[4] Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Par

mar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zheng

dong Zhang, Yonghui Wu, et al. Conformer: Convolution

augmented transformer for speech recognition. In *INTER*

*SPEECH*, 2020. 

[5] TCMR: Beyond Static Features for Temporally Consistent 3D Human Pose and Shape from a Video. CVPR 2021.

[6] Human Mesh Recovery from Multiple Shots. Axiv 2020.

[7] VIBE: Video inference for human body pose and shape estimation. CVPR 2020.

[8] Pose2pose: 3d positional pose-guided 3d rotational pose prediction for expressive 3d human pose and mesh estimation. Axiv 2020.

[9] Pose2mesh: Graph convolutional network for 3d human pose

and mesh recovery from a 2d human pose. In *ECCV*, 2020.

[10] Convolutional mesh regression for single-image hu

man shape reconstruction. In *CVPR*, 2019.

[11] I2l-meshnet: Image

to-lixel prediction network for accurate 3d human pose and

mesh estimation from a single rgb image. In *ECCV*, 2020.

[12] GraphCMR: Convolutional mesh regression for single-image hu

man shape reconstruction. CVPR 2019

[13] HRNet: Deep

high-resolution representation learning for visual recogni

tion. *IEEE Transactions on Pattern Analysis and Machine*

*Intelligence*, 2019.