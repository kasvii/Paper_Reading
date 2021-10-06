# PIFu: Pixel-Aligned Implicit Function for High-Resolution Clothed Human Digitization

## 摘要

本文提出了一个像素对齐的隐函数PIFu，将局部的像素关联到3D物体的全局语境中。使用PIFu训练了一个端到端的网络，能够用来从单帧或者多帧图像推理人体的三维表面和纹理。

## 引入

我们提出了一个新的像素对齐隐函数来表达从单帧或者多帧人体纹理表面推理的三维深度学习。虽然二维的学习已经获得了很多成功，比如语义分割和目标检测，但是三维在全卷积神经网络上面仍然具有很大的挑战：1.体素表示可以应用全卷积，但需要占用很大的内存，限制了表面的精细化。2.全局表达改进了内存的使用效率，但不能保证输入中保留细节。3.基于全局的隐函数虽然能够推理整体形状，但可能不能和输入图像正确对齐。而我们的PIFu可以在像素级别对齐局部特征，在全局对齐整个对象，不需要高内存的使用，人体的形状可以是任意拓扑结构、高度变形或者高度细节的。本文将展示结合局部特征和3D感知隐式表面表达，在甚至只有一张图像的情况下进行精细的重建。

我们训练了一个编码器来学习每个特征向量，图像的每个像素都根据他们的位置加入到全局语境的计算。我们学习一个隐函数来分类一个三维点在表面的里面还是外面，特征向量在空间上将全局三维表面形状与像素对齐，从而能够保留输入图像的局部细节，用来推理看不到的区域。

## 相关工作

### 1. 单视图人体三维数字化

单视图数字化技术需要很大的先验知识。引入深度神经网络来提高位姿和形状的鲁棒性。

- 方法[1] [2]采用部分分割作为输入来提高准确性，尽管能够捕捉人体的测量和运功，但只建立了裸人体。
- [3-5]用于紧身衣，但在更复杂的情况下会失效。
- 无模板方法，比如BodyNet[6]直接学习人体的体素表达，但由于需要很大的内存，所以精细的细节总是被丢弃。
- [7]多视图轮廓推理能够提高内存效率，但是凹形区域很难推测，因此也不能产生精细的细节。

而我们的PIFu能够有效利用内存，并且能够从图像中捕捉精细的细节，并预测每个体素的颜色。

### 2. 多视图三维人体数字化

多视图采集方法是用来产生复杂的人体模型，并简化重建问题，但总是受限于设备和传感器。（1）早期的尝试是从多视图中抠下可见区域，用大量的相机，但是凹面问题难以解决。（2）更准确的几何可以从多视图立体约束或者控制亮度获得。（3）加入了运动信息。虽然多视图技术优于单视图技术，但它们的灵活性和可部署性明显较差。

最近的一些工作：

- [8]训练一个三维卷积LSTM来从任意视图中预测对象的三维体素表达。
- [9]结合任意视图的使用可微的反投影操作。
- [10]使用相似的方法，但需要至少两个视图。以上所有的技术都依赖于体素，需要占用大量内存，并且会丢失高频率的细节。
- [11] [12]基于体积占用场引入深度学习，但至少需要三个视图。

### 3. 纹理预测

三维纹理的推理是从单个或多个图像中预测新视图的视图合成方法。

- [13]介绍了一种可以从前视图来预测后视图的视图合成技术，但无法处理自封闭区域和俯视图。
- [14]恢复检测表面点输出的uv图像，能够推理每个像素的颜色，但精度低。
- [15]在uv参数下直接预测RGB值，但只适用于拓扑结构已知的情况，因此不能用在衣服推测上。

我们提出的方法可以以端到端的方式预测每个顶点的颜色，并可以处理具有任意拓扑结构的曲面。

## PIFu: 像素对齐的隐函数

我们的目标是利用单视图或者多视图输入，来构建强化的三维几何和带纹理的人体，同时保留细节信息。像素对齐隐函数PIFu由全卷积图像编码器 *g* 和一个多层感知机的连续隐函数 *f* 组成：

![img1](./PIFu/1.png)

其中 *x* 是三维点X的二维投影，*z* 是相机坐标系的深度值，*F* 是 *x* 处的图像像素对齐特征，使用双线性采样获得。像素对齐图像特征能够保护局部的细节信息，PIFu隐函数的连续性能够在任意的拓扑结构下产生细节的几何信息，同时不浪费内存。

### Pipeline

![img2](./PIFu/2.png)

### 1. 单视图表面重建

通过最小化下面的均方误差来训练特征编码器 *g* 和隐函数 *PIFu*：

![img2](./PIFu/3.png)

#### 空间采样

训练数据的分辨率对实现隐函数的表达和准确性有重要的作用。跟基于体素的方法不同的是，我们的方法不需要对ground truth的3D mesh进行离散化，用射线追踪算法直接采样即可。采样的策略对最后的重建效果有很大的影响，如果均匀采样的点离表面mesh很远，会给网络带来不必要的权重；而如果采样的点都在表面，会造成过拟合。于是我们结合均匀采样和自适应采样。

首先在表面随机采样，给一个正态分布5厘米的偏移扰动；然后将这些采样点和均匀采样点结合在比例为16的bbox里。

### 2. 纹理推理

虽然纹理推理经常发生在二维参数表面或者视图空间中，但我们的PIFu能够直接在表面结构中推理RGB颜色，使用的是向量场而不是标量场。纹理推理的函数是采样颜色的L1误差：

![img4](./PIFu/4.png)

C是RGB值的ground truth。这个公式能够使图像编码器专注于颜色的推理，而不被看不见的形状、位姿和拓扑结构影响。还引入了一个满足正态分布的偏移，使得颜色能结合表面mesh和周围的三维空间情况推断。

### 3. 多视图立体

<img src="./PIFu/12.png" alt="img12" style="zoom:80%;" />

额外的视图能够提供更多的信息，从而提高重建的准确性。PIFu能够扩展到多视图的情况，将隐函数 *f* 分解为嵌入特征函数 *f1* 和多视图推理函数 *f2*。

## 实验

使用的数据集有：**Render People、BUFF、DeepFashion**

**堆叠沙漏结构**(stacked hourglass architectures)对真实图片有很好的泛化性能；

残差块组成的**CycleGAN**用作纹理推理的图片编码器；

**隐函数**使用多层感知机，层间有图像特征 *F* 和深度 *z* 的跳跃连接；

 **Tex-PIFu**以颜色特征 *Fc* 和用于表面重建的图像特征 *Fv* 作为输入；

**Multi-view PIFu**简单使用中间层的输出作为特征嵌入，并采用平均池化来汇总不同视角的embedding。

### 1. 定量结果

使用了三个指标来评价重建的准确性

- **P2S**：point-to-surface Euclidean distance平均点到表面欧氏距离，从重建表面的顶点到ground truth

- **Chamfer distance**：倒角距离，重建表面和ground truth之间

- **Normal reprojection error**：法向量重投影误差，用来评价重建结果中局部细节的精细度和投影的一致性。

对于重建和ground truth的表面，我们都渲染了他们的法向量地图，并计算他们之间的L2误差。

#### 单视图重建
<img src="./PIFu/5.png" alt="img5" style="zoom: 80%;" />

![img6](./PIFu/6.png)

首先，评价了单视图重建的表面准确度。VRN和IM-GAN和我们的模型都在High-Fidelity Clothed Human数据集上重新进行了训练，而SiCloPr和BodyNet用的是他们的训练模型。尽管单视图的输入具有尺度不确定性，但评估的时候是在已知尺度因子下执行的。和同样是隐函数的IM-GAN相比，我们的方法输出像素对齐的高精度表面重建结果，能够捕获发型和衣服的褶皱。和VRN相比，虽然VRN和我们的图像编码器共享同样的网络架构，但是隐函数的更高表达方式让我们能够实现更高的保真度。

<img src="./PIFu/7.png" alt="img7" style="zoom:80%;" />

接着评价了纹理推理。SiCloPe从二维图片来推理后视图，并将其和前视图的纹理缝合在一起，这会面临投影变形和轮廓伪影的问题。而我们的方法直接从表面mesh中推理纹理，不存在投影伪影的问题。

#### 多视图重建

<img src="./PIFu/8.png" alt="img8" style="zoom:80%;" />

<img src="./PIFu/9.png" alt="img9" style="zoom:80%;" />

<img src="./PIFu/10.png" alt="img10" style="zoom:80%;" />

图8说明，随着视图的增加，我们的方法能够增量式地精细化几何和纹理。

### 2. 定性结果

![img11](./PIFu/11.png)

## 总结

### 本文的贡献

- 提出了一个像素对齐的隐函数，能够在像素水平将输入图像和三维物体对齐，能够从单帧图像推理人体三维几何和着衣纹理；
- 能够推理出高可信度的不可见区域的三维几何，并保留细节，同时不需要大量的内存；
- 是第一个可以为任意拓扑形状绘制纹理的方法，可以直接从表面预测看不见区域和凹区域和边缘区域的颜色；
- 同时还拓展到多视图重建，随着视图的增加，细节信息也更加丰富。

### 未来的工作

#### 1. 提高重建精度

- 使用生成对抗网络GAN
- 提高输入图像的分辨率

#### 2. 尺度因子的推理

#### 3. 遮挡问题

在部分遮挡的情况下对人体进行推理和重建

## 关键词

iso-surface - 等值面

Marching Cube algorithm - 标记立方体算法

3D occupancy field - 三维占用场

bilinear sampling - 双线性采样

mean squared error - 均方误差

spatially aligned - 空间对齐

spatial sampling - 空间采样

ray tracing algorithm - 射线追踪算法

water-tight mesh 防水网格

normal distribution - 正态分布

arbitrary topology - 任意拓扑结构

self-occlusion - 自遮挡？？？

stacked hourglass architectures - 堆叠沙漏结构

CycleGAN 

multi-layer perceptron - 多层感知机

average pooling - 平均池化

point-to-surface Euclidean distance - 点到表面欧氏距离(P2S)

chamfer distance - 倒角距离

normal reprojection error -  法向量重投影误差

projection artifacts - 投影伪影

projection distortion - 投影变形

## 数据集

**RenderPeople**：https://renderpeople.com/3d-people.

**BUFF**：Chao Zhang, Sergi Pujades, Michael Black, and Gerard Pons Moll. Detailed, accurate, human shape estimation from clothed 3D scan sequences. In *IEEE Conference on Computer* *Vision and Pattern Recognition*, pages 4191–4200, 2017.

**DeepFasion**：Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang. Deepfashion: Powering robust clothes recognition and retrieval with rich annotations. In *IEEE Conference on* *Computer Vision and Pattern Recognition*, pages 1096–1104, 2016.

## 参考

[1] Christoph Lassner, Javier Romero, Martin Kiefel, Federica Bogo, Michael J Black, and Peter V Gehler. Unite the people: Closing the loop between 3d and 2d human representations. In *IEEE Conference on Computer Vision and Pattern Recognition*, pages 6050–6059, 2017.

[2] Mohamed Omran, Christoph Lassner, Gerard Pons-Moll, Peter V. Gehler, and Bernt Schiele. Neural body fifitting: Unifying deep learning and model-based human pose and shape estimation. In *International Conference on 3D Vision*, pages 484–494, 2018.

[3] Thiemo Alldieck, Marcus Magnor, Weipeng Xu, Christian Theobalt, and Gerard Pons-Moll. Detailed human avatars from monocular video. In *International Conference on 3D* *Vision*, pages 98–109, 2018.

[4] Thiemo Alldieck, Marcus Magnor, Bharat Lal Bhatnagar, Christian Theobalt, and Gerard Pons-Moll. Learning to reconstruct people in clothing from a single RGB camera. In *IEEE* *Conference on Computer Vision and Pattern Recognition*, pages 1175–1186, 2019.

[5] Chung-Yi Weng, Brian Curless, and Ira Kemelmacher Shlizerman. Photo wake-up: 3d character animation from a single photo. *arXiv preprint arXiv:1812.02246*, 2018.

[6] Gul Varol, Duygu Ceylan, Bryan Russell, Jimei Yang, Ersin Yumer, Ivan Laptev, and Cordelia Schmid. BodyNet: Volumetric inference of 3D human body shapes. In *European* *Conference on Computer Vision*, pages 20–36, 2018.

[7] Ryota Natsume, Shunsuke Saito, Zeng Huang, Weikai Chen, Chongyang Ma, Hao Li, and Shigeo Morishima. Siclope: Silhouette-based clothed people. In *IEEE Conference on* *Computer Vision and Pattern Recognition*, pages 4480–4490, 2019.

[8] Christopher B Choy, Danfei Xu, JunYoung Gwak, Kevin Chen, and Silvio Savarese. 3d-r2n2: A unifified approach for single and multi-view 3d object reconstruction. In *European* *Conference on Computer Vision*, pages 628–644, 2016.

[9] Abhishek Kar, Christian Hane, and Jitendra Malik. Learning a multi-view stereo machine. In *Advances in Neural* *Information Processing Systems*, pages 364–375, 2017.

[10] Mengqi Ji, Juergen Gall, Haitian Zheng, Yebin Liu, and Lu Fang. Surfacenet: An end-to-end 3d neural network for multiview stereopsis. In *IEEE Conference on Computer* *Vision and Pattern Recognition*, pages 2307–2315, 2017.

[11] Zeng Huang, Tianye Li, Weikai Chen, Yajie Zhao, Jun Xing, Chloe LeGendre, Linjie Luo, Chongyang Ma, and Hao Li. Deep volumetric video from very sparse multi-view performance capture. In *European Conference on Computer* *Vision*, pages 336–354, 2018.

[12] Andrew Gilbert, Marco Volino, John Collomosse, and Adrian Hilton. Volumetric performance capture from minimal camera viewpoints. In *European Conference on Computer Vision*, pages 566–581, 2018.

[13] Ryota Natsume, Shunsuke Saito, Zeng Huang, Weikai Chen, Chongyang Ma, Hao Li, and Shigeo Morishima. Siclope: Silhouette-based clothed people. In *IEEE Conference on* *Computer Vision and Pattern Recognition*, pages 4480–4490, 2019.

[14] Natalia Neverova, Riza Alp Guler, and Iasonas Kokkinos. Dense pose transfer. In *European Conference on Computer Vision*, pages 123–138, 2018.

[15] Angjoo Kanazawa, Shubham Tulsiani, Alexei A. Efros, and Jitendra Malik. Learning category-specifific mesh reconstruction from image collections. In *European Conference on* *Computer Vision*, pages 371–386, 2018.

