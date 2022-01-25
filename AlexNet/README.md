# AlexNet: ImageNet Classifification with Deep Convolutional

## 摘要

We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of fifive convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a fifinal 1000-way softmax. To make training faster, we used non-saturating neurons and a very effificient GPU implementation of the convolution operation. To reduce overfifitting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.

我们训练了一个大型的深度卷积神经网络，将ImageNet-2010比赛中的120万张高分辨率图像分类为1000个不同的类型。在测试集上，我们实现了前1和前5的错误率分别为37.5%和17.0%，比之前的SOTA方法好得多。这个神经网络有6000万个参数和65万个神经元，包括5层神经网络，其中一些后面跟着最大池化和3个全连接层，最后是1000路的softmax。为了使训练速度更快，我们使用非饱和神经元和一个非常有效的GPU卷积操作。为了减少全连接层中的过拟合，我们采用了一种最近开发的被称为“dropout”的正则化方法，该方法被证明是非常有效的。我们还在ILSVRC-2012竞赛中输入了该模型的一个变体，获得了15.3%的前5名，而第二名的错误率为26.2%。

![1](AlexNet/1.png)





## 关键词

**top-1 and top-5 error rates** - 前1和前5误差率：一个图片经过网络，得到预测类别的概率，如果概率前1或者前5（top-1 or top-5）中包含正确答案，即认为正确。

**label-preserving transformations** - 标签保留的转换，一种减少过拟合的方式，在不影响图像标签的前提下，对图像进行变换，以达到数据增强的目的。

**目标分割、目标检测、目标识别、目标跟踪**：

- **目标分割** （Target Segmentation）：任务是把目标对应部分分割出来。

  像素级的前景与背景的分类问题，将背景剔除。

  举例：（以对视频中的小明同学进行跟踪为例，列举处理过程）

  第一步进行目标分割，采集第一帧视频图像，因为人脸部的肤色偏黄，因此可以通过颜色特征将人脸与背景分割出来。

- **目标检测**（Target Detection）：定位目标，确定目标位置和大小。检测目标的有无。

  检测有明确目的性，需要检测什么就去获取样本，然后训练得到模型，最后直接去图像上进行匹配，其实也是识别的过程。

  举例：第二步进行目标识别，分割出来后的图像有可能不仅仅包含人脸，可能还有部分环境中颜色也偏黄的物体，此时可以通过一定的形状特征将图像中所有的人脸准确找出来，确定其位置及范围。

- **目标识别**（Target Recognition）：定性目标，确定目标的具体模式（类别）。

  举例：第三步进行目标识别，将图像中的所有人脸与小明的人脸特征进行对比，找到匹配度最好的，从而确定哪个是小明。

- **目标跟踪**（Target Tracking）：追踪目标运动轨迹。

  举例：第四步进行目标跟踪，之后的每一帧就不需要像第一帧那样在全图中对小明进行检测，而是可以根据小明的运动轨迹建立运动模型，通过模型对下一帧小明的位置进行预测，从而提升跟踪的效率。

**feedforward neural network** - 前馈神经网络（[一文理清深度学习前馈神经网络](https://www.cnblogs.com/samshare/p/11801806.html)）

**防止过拟合方法**：Data Augmentation（数据增广）、Regularization（正则化）、Model Ensemble（模型集成）、Dropout等等



## 数据集

**NORB** - 该数据库致力于从外形进行3D对象识别的试验，其中包含50个玩具图像，分属于5种范畴：四肢动物、人像、飞机、卡车和汽车等

**Caltech-101/256** ：

- Caltech-101数据集（用作101类图像分类）

  这个数据集包含了101类的图像，每类大约有40~800张图像，大部分是50张/类；在2003年由lifeifei收集，每张图像的大小大约是300x200。

- Caltech-256数据集（用作256类的图像分类）

  此数据集和Caltech-101相似，包含了30,607张图像。

**CIFAR-10/100**：

- CIFAR-10数据集

  由10个类的60000个32x32彩色图像组成，每个类有6000个图像。有50000个训练图像和10000个测试图像。

- CIFAR-100数据集

  这个数据集就像CIFAR-10，除了它有100个类，每个类包含600个图像。每类各有500个训练图像和100个测试图像。CIFAR-100中的100个类被分成20个超类。每个图像都带有一个“精细”标签（它所属的类）和一个“粗糙”标签（它所属的超类）

**MNIST**：

由来自 250 个不同人手写的数字构成, 其中 50% 是高中学生, 50% 来自人口普查局 (the Census Bureau) 的工作人员. 测试集(test set) 也是同样比例的手写数字数据。由60000个训练样本和10000个测试样本组成，每个样本都是一张28 * 28像素的灰度手写数字图片。

**LabelMe**：

- LabelMe 12-50k 数据集是一个物体识别数据集，总共包含 50,000 张 JPEG 格式的图片，其中 40,000 张为训练数据，10,000张为测试数据，图像均从 LabelMe 网站中提取得到。每张图像分辨率为 256x256 大小。其中 50% 的图片在中心位置有一个物体，该物体属于类别总数为12类中的一类。其余 50% 图片为随机选取的图片中的一块随机区域。

**ImageNet** - 是一项持续的研究工作，旨在为世界各地的研究人员提供易于访问的图像数据库。目前ImageNet中总共有14,197,122幅图像，总共分为21,841个类别(synsets)



## 参考文献

[1] Angjoo Kanazawa, Jason Y. Zhang, Panna Felsen, and Jitendra Malik. Learning 3D human dynamics from video. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

[2] Yu Sun, Yun Ye, Wu Liu, Wenpeng Gao, Yili Fu, and Tao Mei. Human mesh recovery from monocular images via a skeleton-disentangled representation. In *International Conference on Computer Vision*, 2019.

[3] Angjoo Kanazawa, Michael J. Black, David W. Jacobs, and Jitendra Malik. End-to-end recovery of human shape and pose. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2018.

[4] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In *Advances in Neural Information Processing*, 2014.

[5] Dushyant Mehta, Oleksandr Sotnychenko, Franziska Mueller, Weipeng Xu, Srinath Sridhar, Gerard Pons-Moll, and Christian Theobalt. Single-shot multi-person 3D pose estimation from monocular RGB. In *International Conference on 3DVision*, 2018.

[6] Dushyant Mehta, Srinath Sridhar, Oleksandr Sotnychenko, Helge Rhodin, Mohammad Shafifiei, Hans-Peter Seidel, Weipeng Xu, Dan Casas, and Christian Theobalt. VNect: Real-time 3D human pose estimation with a single RGB camera. In *SIGGRAPH*, July 2017.

[7] Federica Bogo, Angjoo Kanazawa, Christoph Lassner, Peter Gehler, Javier Romero, and Michael J. Black. Keep it SMPL: Automatic estimation of 3D human pose and shape from a single image. In *European Conference on Computer Vision*, 2016. 

[8] Anurag Arnab, Carl Doersch, and Andrew Zisserman. Exploiting temporal context for 3D human pose estimation in the wild. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2019. 

[9] Angjoo Kanazawa, Jason Y. Zhang, Panna Felsen, and Jitendra Malik. Learning 3D human dynamics from video. In*IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

[10] Yu Sun, Yun Ye, Wu Liu, Wenpeng Gao, Yili Fu, and Tao Mei. Human mesh recovery from monocular images via a skeleton-disentangled representation. In *International Conference on Computer Vision*, 2019.
