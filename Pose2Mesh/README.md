# Pose2Mesh: Graph Convolutional Networkfor 3D Human Pose and Mesh Recovery from a 2D Human Pose

![1](./Pose2Mesh/1.png)

[TOC]



## 摘要

最近大多数基于深度学习的三维人体姿态和网格估计方法都是从输入图像中回归出人体网格模型的姿态和形状参数，如SMPL和MANO。这些方法的**第一个缺点是对图像外观的过拟合**，因为训练数据的可控设置和测试时的in-the-wild数据的域差异。第二个缺点是，由于三维旋转的表达问题，姿态参数的估计相当具有挑战性。为了克服上述缺点，我们提出了Pose2Mesh，一种新的**基于图卷积神经网络**(GraphCNN)的系统，它**直接从二维人体姿态估计人体网格顶点**的三维坐标。二维人体姿态作为输入，提供了基本的人体关节信息，而没有图像外观。此外，该系统避免了表示问题，同时利用**GraphCNN以从粗到细**的方式充分利用网格拓扑。我们表明，我们的Pose2Mesh在各种基准数据集上显著优于以前的三维人体姿态和网格估计方法。

## 引入

三维人体姿态和网格估计的目的是**同时恢复三维人体关节和网格顶点的位置**。

它的挑战在于：深度和尺度的模糊性、复杂的人体和手的关节。

学习的方法主要有两种：基于模型的方法训练网络预测模型（SMPL、MANO）参数，并通过解码生成人网格，和无模型方法直接回归三维人网格的坐标。两类方法都通过**将输出网格与关节回归矩阵相乘，来计算三维人体姿态**。

目前的方法有两个不足：

- 1. 训练集设置得太好了，有单一的北京和简单的衣着，没有图像过拟合的问题，与in-the-wild图像有很大的不同。

- 2. 人类网格模型的姿态参数可能不是一个适当的回归目标。例如，SMPL姿态参数表示轴角的三维旋转，这可能会出现非唯一的问题（即周期性）。虽然在许多工作中，[23,29,42]试图通过使用旋转矩阵作为预测目标来避免周期性，但它仍然存在一个非最小的表示问题。

我们提出的Pose2Mesh，是一个从二维人体姿态中恢复三维人体姿态和网格的图卷积网络。与现有的方法相比，它有两个优点：

- 1. 输入的二维人体姿态使该系统没有与图像外观相关的过拟合，同时提供了关于人体关节的基本几何信息。此外，二维的人体姿态可以从in-the-wild图像中准确估计；

- 2. 无模型方法，直接回归网格顶点：Pose2Mesh在利用了人形网格拓扑结构（即面信息和边缘信息）时，避免了姿态参数的表示问题。它利用由网格拓扑构造的图的图卷积神经网络(GraphCNN)直接对网格顶点的三维坐标进行回归。

我们设计了一个级联结构，包括PoseNet和MeshNet：

- PoseNet将2D人体姿势提升为3D人体姿势
- MeshNet同时采用2D和3D的人体姿态，利用从粗到细的方式来估计三维人体网格。

在前向传播过程中，网格特征首先以**粗分辨率**进行处理，然后逐渐上采样到**精细分辨率**。如图是pipline。

![2](Pose2Mesh/2.png)

### 贡献

- 我们提出了一种新的系统，Pose2Mesh，它可以从**二维人体姿态**中恢复**三维人体姿态和网格**。它对图像外观没有过拟合，因此可以很好地推广到in-the-wild数据上
- 我们的Pose2网格使用GraphCNN直接回归人体网格的三维坐标。它避免了模型参数的表示问题，并利用了预定义的网格拓扑

## 相关工作

### 三维人体姿态估计

根据输入类型，三维人体姿态估计方法可以分为两类：基于图像的方法和基于二维姿态的方法。

- 基于图像的方法以RGB图像作为输入，输出三维人体姿态估计；
- 基于二维姿态的方法，是将二维姿态提升到三维空间。

我们的工作是利用基于二维姿态的方法，使得Pose2Mesh在不同域下有更好的鲁棒性

### 三维人体和手姿态和网格估计

人体网格的估计也有两种方法：基于模型和无模型

- 基于模型的有：NBF（Neural Body Fitting, human part segmentation, 3DV 2018）、SPIN（self-improving, ICCV 2019）、HMR（adversarial loss, CVPR 2018）
- 无模型的方法：GraphCMR （GraphCNN, CVPR 2019 Oral）、I2L-MeshNet（heatmap representation, ECCV 2020）

### 用于处理mesh的GraphCNN

许多方法**将mesh作为一种图结构**，并使用**GraphCNN**来处理，因为与简单的堆叠全连接层相比，它可以完全利用**网格拓扑**。利用GraphCNN的方法有：

- Pixel2mesh（ECCV 2018）：采用GraphCNN从粗到细地学习从初始椭球体网格到目标对象网格的变形（迭代优化的方法）
- Feastnet（CVPR 2018）：提出了一个新的GraphCNN算子
- ConvMeshAE（ECCV 2018）：提出了基于GraphCNN的VAE，以分层的方式学习人脸网格的潜在空间（VAE，变分自动编码器）

## PoseNet

### 对输入二维姿态加入误差

PoseNet利用二维姿态输入来评估三维姿态。其中，估计的二维姿态往往有错误，于是我们的ground truth的二维姿态上**加入了真实的误差**，合成$P^{2D}$。

### 二维输入姿态归一化

处理：从$P^{2D}$中减去平均值，再除以标准差，得到$\overline{P}^{2D}$。平均值和标准差分别代表二维位置和尺度，这种归一化是必要的，因为$P^{3D}$独立于二维姿态的尺度和位置。

### 网络结构

输入归一化的二维姿态$\overline{P}^{2D}$，通过全连接层转换为4096维特征向量，再经过两个残差块，残差块的输出特征向量，通过全连接层转换为3J（J是关节数）维的向量，表示$P^{3D}$。

### 损失函数

最小化预测三维位姿$P^{3D}$和ground truth的L1距离来训练PoseNet。
$$
L_{\text {pose }}=\left\|\mathbf{P}^{3 \mathrm{D}}-\mathbf{P}^{3 \mathrm{D}^{*}}\right\|_{1}
$$
*代表ground truth

## MeshNet

将$\overline{P}^{2D}$和$P^{3D}$连接在一起成为$\mathbf{P} \in \mathbb{R}^{J \times 5}$，然后估计三维网格$\mathbf{M} \in \mathbb{R}^{V \times 3}$，V是网格顶点数。MeshNet的设计是基于切比雪夫光谱图卷积Chebysev spectral graph convolution。

### 图结构

图$\mathcal{G}_{\mathrm{P}}=\left(\mathcal{V}_{\mathrm{P}}, A_{\mathrm{P}}\right)$，前者是关节点集合，后者是邻接矩阵[0,1]，0表示关节不连接，1表示关节连接。

### 光谱图卷积

然后MeshNet在 $\mathcal{G}_{\mathrm{P}}$ 执行光谱图卷积

### 从粗到细的网格上采样

![3](Pose2Mesh/3.png)

逐渐从$\mathcal{G}_{\mathrm{P}}$上采样到M，$\mathcal{G}_{\mathrm{M}}=\left(\mathcal{V}_{\mathrm{M}}, A_{\mathrm{M}}\right)$，前者是网格顶点，后者是网格顶点的邻接矩阵，定义网格的边缘。然后对$\mathcal{G}_M$利用图粗糙化得到不同的精度，如图2。

在前向传播的过程中，MeshNet首先利用reshape和全连接层上采样$\mathcal{G}_P$到最粗糙的网格图$\mathcal{G}_M$。然后执行光谱图卷积。
$$
F_{\text {out }}=\sum_{k=0}^{K-1} T_{k}\left(\tilde{L_{\mathrm{M}}}^{c}\right) F_{\mathrm{in}} \Theta_{k}
$$
上采样的定义：
$$
F_{c}=\psi\left(F_{c+1}^{T}\right)^{T}
$$
$\psi: \mathbb{R}^{f_{c+1} \times \mathcal{V}_{\mathrm{M}}^{c+1}} \rightarrow \mathbb{R}^{\mid f_{c+1} \times \mathcal{V}_{\mathrm{M}}^{c}}$ 表示最近邻上采样函数

利用平衡二叉树的数据结构，$\mathcal{G}_{\mathrm{M}}^{c+1}$中的第i个vertex是$\mathcal{G}_{\mathrm{M}}^{c}$第2i和2i-1的父节点。

<img src="Pose2Mesh/4.png" alt="4" style="zoom:100%;" />

在每个精度之间，加入了残差连接。

### 损失函数

用了四个损失函数来训练MeshNet

- 顶点坐标损失

  最小化预测三维网格顶点和ground truth的L1距离：
  $$
  L_{\text {vertex }}=\left\|\mathbf{M}-\mathbf{M}^{*}\right\|_{1}
  $$

- 关节坐标损失

  最小化Mesh回归的三维姿态和ground truth的L1距离，来训练mesh顶点和关节位置对齐。
  $$
  L_{\text {joint }}=\left\|\mathcal{J} \mathbf{M}-\mathbf{P}^{3 \mathrm{D}^{*}}\right\|_{1}
  $$

- 表面法向量损失

  监督mesh的法向量和ground truth的一致，来提高表面的光滑性和局部细节。

  f和nf分别是mesh和ground truth的一个三角面及其法向量
  $$
  L_{\text {normal }}=\sum_{f} \sum_{\{i, j\} \subset f}\left|\left\langle\frac{\mathbf{m}_{i}-\mathbf{m}_{j}}{\left\|\mathbf{m}_{i}-\mathbf{m}_{j}\right\|_{2}}, n_{f}^{*}\right\rangle\right|
  $$

- 表面边缘损失

  这个损失函数对手、脚和嘴等密集顶点的光滑性很有效。
  $$
  L_{\text {edge }}=\sum_{f} \sum_{\{i, j\} \subset f}\left|\left\|\mathbf{m}_{i}-\mathbf{m}_{j}\right\|_{2}-\left\|\mathbf{m}_{i}^{*}-\mathbf{m}_{j}^{*}\right\|_{2}\right|
  $$

- 总损失函数
  $$
  L_{\text {mesh }}=\lambda_{\mathrm{v}} L_{\text {vertex }}+\lambda_{\mathrm{j}} L_{\text {joint }}+\lambda_{\mathrm{n}} L_{\text {normal }}+\lambda_{\mathrm{e}} L_{\text {edge }},
  $$

*λ*v = 1*, λ*j = 1*, λ*n = 0*.*1*,* and *λ*e = 20

## 实施细节

- 优化器：Rmsprop
- 学习率：$10^{-3}$，30个epoch降低10倍
- 

## 专业词汇

joint regression matrix - 关节回归矩阵

pre-defifined mesh topology - 预定义的网络拓扑？？？

stacked fully-connected layers - 堆叠全连接层

mesh topology - 网格拓扑

spectral graph convolution - 光谱图卷积

adjacency matrix - 邻接矩阵

 *K*-order polynomial - K接多项式

graph coarsening - 图粗糙化

a balanced binary tree - 平衡二叉树 

triangle face - 三角面