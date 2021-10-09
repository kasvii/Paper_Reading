# DeepCap: Monocular Human Performance Capture Using Weak Supervision

## 关键词

space-time coherent geometry - 时空几何一致性

weakly supervised manner- 弱监督方式

pose estimation - 位姿估计

non-rigid surface deformation - 非刚性表面变形

multi-view marker-less method - 多视图无标记方法

monocular human modeling approaches - 单目人体重建方法

articulated template - 关节模板

## 摘要

人体三维重建在虚拟现实和增强现实有着越来越多的使用，之前的许多工作要么需要昂贵的多视图设备，要么没有利用帧到帧的相关性来重建稠密时空一致性几何。我们的方法基于多视图监督训练了一个弱监督的方式，完全消除了对三维ground truth标注的需要。该工作分解为两个独立的网络，分别是位姿估计和非刚性表面变形。在定性和定量实验中展现了SOTA的性能和鲁棒性。

## 引入

人体三维重建的作用：

- 电影
- 游戏
- 生成个性化的动态虚拟
- 混合现实

目前大部分的单目方法只捕获了结构的运动，比如手和稀疏的面部表情，但是用在稠密全身（皮肤或着衣）的变形的单目跟踪仍然处于开始阶段。

许多多视图无标记的方法已经获得了很好的成果，但是需要多相机的工作室（比如绿幕），限制了在现实中本地摄影条件下的使用。

目前的单目人体建模方法已经在衣服、发型和面部细节有很好的效果。[1] [2] 直接回归了体素，[3]回归了连续的表面占用。因为这些预测都是像素对齐的，所以重建的细节很好，但四肢经常跟丢。重建运动分解为结构和非刚性变形，能够防止计算图风格控制超过重建，更有应用价值。更重要的是，表面顶点并不是总能被跟踪到，重建不是时空关联的。另外的方法是使用关节模板，虽然减少了肢体丢失，增加了控制，但是没有捕获到运动和表面变形。

目前的单目人体捕获方法是通过密集跟踪表面变形。利用基于深度学习的稀疏关键点检测，再执行复杂的模板拟合。因此，他们只能非刚性地拟合输入图像，并有很大的不稳定性。而我们提出了第一个在单前馈通道下，联合推理关节和非刚性三维形变参数的方法。核心算法是融合完全可微的网格模板，包括位姿和嵌入式变形图，的卷积神经网络。对于一个单帧图像，我们的网络预测了骨架位姿，变形图中每个节点的旋转和平移参数。与隐式表示形成鲜明对比，我们的方法能够实时跟踪表面顶点，能够用于增加语义、贴图和渲染。除此之外，我们模型还有一个优点是不会丢失肢体，甚至在遮挡或者图像外运动的情况。

## 数据集



## 参考文献

[1] V. Gabeur, J.-S. Franco, X. Martin, C. Schmid, and G. Rogez. Moulding humans: Non-parametric 3d human shape estimation from single images. In *Proceedings of the IEEE International Conference on Computer Vision*, pages 2232–2241, 2019. 

[2] Z. Zheng, T. Yu, Y. Wei, Q. Dai, and Y. Liu. Deephuman: 3d human reconstruction from a single image. *CoRR*, abs/1903.06473, 2019. 1, 2, 6, 7, 8

[3] S. Saito, Z. Huang, R. Natsume, S. Morishima, A. Kanazawa, and H. Li. Pifu: Pixel-aligned implicit function for high-resolution clothed human digitization. *CoRR*, abs/1905.05172, 2019. 1, 2, 6, 7, 8