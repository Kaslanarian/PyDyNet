# PyDyNet：Neuron Network(DNN, CNN, RNN, etc) implementation using Numpy based on Autodiff

前作：[PyNet: Use NumPy to build neuron network](https://github.com/Kaslanarian/PyNet)。在那里我们基于求导规则实现了全连接网络。在这里，我们向当今的深度学习框架看齐，实现属于自己的DL框架。

**PyDyNet已被多个技术公众号和社区分享**：[居然用Numpy实现了一个深度学习框架](https://segmentfault.com/a/1190000042108301).

[![Downloads](https://pepy.tech/badge/pydynet)](https://pepy.tech/project/pydynet)
[![Downloads](https://static.pepy.tech/personalized-badge/pydynet?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/pydynet)
![](https://img.shields.io/pypi/l/pydynet)
![](https://img.shields.io/pypi/implementation/numpy)
![](https://img.shields.io/github/stars/Kaslanarian/PyDyNet?style=social)
![](https://img.shields.io/github/forks/Kaslanarian/PyDyNet?style=social)

## Update

- 5.10: ver 0.0.1 修改损失函数的定义方式：加入reduction机制，加入Embedding;
- 5.15: ver 0.0.2 重构了RNN, LSTM和GRU，支持双向;
- 5.16: ver 0.0.2 允许PyDyNet作为第三方库安装；开始手册的撰写(基于Sphinx).
- 5.29: ver 0.0.3 加入了Dataset和Dataloader，现在可以像PyTorch一样定义数据集和分割数据集，具体参考[data.py](/pydynet/data.py)中的`train_loader`函数；
- 5.30: ver 0.0.3 将一维卷积算法退化成基于循环的im2col，新版本NumPy似乎不是很支持strided上数组的魔改；
- 7.22: ver 0.0.4 增加了Module类和Parameter类，将模块重组、增加多种Pytorch支持的初始化方式；正在撰写新的Manual；
- 7.28: ver 0.0.6 加入no_grad方法，可以像pytorch一样禁止自动微分，比如`@no_grad()`和`with no_grad()`，详见[autograd.py](/pydynet/autograd.py);
- 8.09: ver 0.0.7 基于[cupy](https://cupy.dev/)，PyDyNet现在可以使用显卡加速训练，用法与PyTorch一致，详见[tests](./tests)中`cu*.py`；
- ...

## Overview

PyDyNet也是纯NumPy(0.0.6版本后加入CuPy，其用法和NumPy一致)实现的神经网络，语法受PyTorch的启发，大致结构如下：

```mermaid
graph BT
   N ----> ds(Dataset) ----> Data(DataLoader)
   N(numpy.ndarray/cupy.ndarray) --> A(Tensor)
   A --Eager execution--> B(Basic operators: add, exp, etc)
   B --> E(Mechanism: Dropout, BN, etc)
   E --> D
   B --> C(Complex operators: softmax, etc)
   C --> E
   B --> D(Base Module:Linear, Conv2d, etc)
   C --> D
   B --Autograd--> A
   N ----> GD(Optimizer:SGD, Adam, etc)
   D --> M(Module:DNN, CNN, RNN, etc)
   M --> Mission(PyDyNet)
   Data --> Mission
   GD --> Mission
```

文件结构

```bash
pydynet
├── __init__.py
├── autograd.py       # 微分控制模块
├── cuda.py           # cuda功能模块
├── data.py           # 数据集模块
├── nn                # 神经网络模块
│   ├── __init__.py   
│   ├── functional.py # 函数类
│   ├── init.py       # 初始化模块
│   ├── modules
│   │   ├── __init__.py
│   │   ├── activation.py # 激活函数
│   │   ├── batchnorm.py  # BN
│   │   ├── conv.py       # 卷积和池化
│   │   ├── dropout.py    # dropout
│   │   ├── linear.py     # 线性层
│   │   ├── loss.py       # 损失函数类
│   │   ├── module.py     # Module基类，包括Sequential
│   │   └── rnn.py        # RNN
│   └── parameter.py      # 参数化类
├── optim.py              # 优化器类
└── tensor.py             # 张量类
```

我们实现了：

1. 将NumPy数组包装成具有梯度等信息的张量(Tensor):

   ```python
   from pydynet import Tensor

   x = Tensor(1., requires_grad=True)
   print(x.data) # 1.
   print(x.ndim, x.shape, x.is_leaf) # 0, (), True
   ```

2. 将NumPy数组的计算(包括数学运算、切片、形状变换等)抽象成基础算子(Basic operators)，并对部分运算加以重载：

   ```python
   from pydynet import Tensor
   import pydynet.functional as F

   x = Tensor([1, 2, 3])
   y = F.exp(x) + x
   z = F.sum(x)
   print(z.data) # 36.192...
   ```

3. 手动编写基础算子的梯度，实现和PyTorch相同的动态图自动微分机制(Autograd)，从而实现反向传播

   ```python
   from pydynet import Tensor
   import pydynet.functional as F

   x = Tensor([1, 2, 3], requires_grad=True)
   y = F.log(x) + x
   z = F.sum(y)

   z.backward()
   print(x.grad) # [2., 1.5, 1.33333333]
   ```

4. 基于基础算子实现更高级的算子(Complex operators)，它们不再需要手动编写导数：

   ```python
   def simple_sigmoid(x: Tensor):
       return 1 / (1 + exp(-x))
   ```

5. 实现了Mudule，包括激活函数，损失函数等，从而我们可以像下面这样定义神经网络，损失函数项：

   ```python
   import pydynet.nn as nn
   import pydynet.functional as F

   n_input = 64
   n_hidden = 128
   n_output = 10

   class Net(nn.Module):
       def __init__(self) -> None:
           super().__init__()
           self.fc1 = nn.Linear(n_input, n_hidden)
           self.fc2 = nn.Linear(n_hidden, n_output)

       def forward(self, x):
           x = self.fc1(x)
           x = F.sigmoid(x)
           return self.fc2(x)

   net = Net()
   loss = nn.CrossEntropyLoss()
   l = loss(net(X), y)
   l.backward()
   ```

6. 实现了多种优化器(`optimizer.py`)，以及数据分批的接口(`dataloader.py`)，从而实现神经网络的训练；其中优化器和PyTorch一样支持权值衰减，即正则化；
7. Dropout机制，Batch Normalization机制，以及将网络划分成训练阶段和评估阶段；
8. 基于im2col高效实现Conv1d, Conv2d, max_pool1d和max_pool2d，从而实现CNN；
9. 支持多层的**双向**RNN，LSTM和GRU；
10. 实现了PyTorch中的Dataset类、DataLoader类，从而将批数据集封装成迭代器；
11. 多种初始化方式，包括Kaiming和Xavier；
12. 基于cupy实现了显卡计算和训练：

   ```python
   from pydynet import Tensor
   
   x = Tensor([1., 2., 3.], device='cuda')
   y = Tensor([1., 2., 3.], device='cuda')
   z = (x * y).sum()

   w = Tensor([1., 2., 3.]) # CPU上的Tensor
   x * w # 报错
   ```

13. ...

## Install

```bash
pip install pydynet
```

或本地安装

```bash
git clone https://github.com/Kaslanarian/PyDyNet
cd PyDyNet
python setup.py install
```

安装成功后就可以运行下面的例子

## Example

[tests](./tests)中是一些例子。

### AutoDiff

[autodiff.py](tests/autodiff.py)利用自动微分，对一个凸函数进行梯度下降：

![ad](src/autodiff.png)

### DNN

[DNN.py](tests/DNN.py)使用全连接网络对`sklearn`提供的数字数据集进行分类，训练参数

- 网络结构：Linear(64->64) + Sigmoid + Linear(64->10)；
- 损失函数：Cross Entropy Loss；
- 优化器：Adam(lr=0.01)；
- 训练轮次：50；
- 批大小(Batch size)：32.

训练损失，训练准确率和测试准确率：

<img src="src/DNN.png" alt="dnn" style="zoom:67%;" />

### CNN

[CNN.py](tests/CNN.py)使用三种网络对`fetch_olivetti_faces`人脸(64×64)数据集进行分类并进行性能对比：

1. Linear + Sigmoid + Linear;
2. Conv1d + MaxPool1d + Linear + ReLU + Linear;
3. Conv2d + MaxPool2d + Linear + ReLU + Linear.

其余参数相同：

- 损失函数：Cross Entropy Loss；
- 优化器：Adam(lr=0.01)；
- 训练轮次：50；
- 批大小(Batch size)：32.

学习效果对比：

<img src="src/CNN.png" alt="cnn" style="zoom:67%;" />

## Droput & BN

[dropout_BN.py](tests/dropout_BN.py)使用三种网络对`fetch_olivetti_faces`人脸(64×64)数据集进行分类并进行性能对比：

1. Linear + Sigmoid + Linear;
2. Linear + Dropout(0.05) + Sigmoid + Linear;
3. Linear + BN + Sigmoid + Linear.

其余参数相同：

- 损失函数：Cross Entropy Loss；
- 优化器：Adam(lr=0.01)；
- 训练轮次：50；
- 批大小(Batch size)：32.

学习效果对比：

<img src="src/dropout_BN.png" alt="BN" style="zoom:67%;" />

## RNN

[RNN.py](tests/RNN.py)中是一个用双向单层GRU对`sklearn`的数字图片数据集进行分类：

<img src="src/RNN.png" alt="RNN" style="zoom:67%;" />

## cuda相关

[cuDNN.py](tests/cuDNN.py), [cuCNN.py](tests/cuCNN.py), [cuDropoutBN.py](tests/cuDropoutBN.py), [cuRNN.py](tests/cuRNN.py)分别是上面四种网络的cuda版本，并对网络进行了相应的修改，主要是介绍如何使用PyDyNet的显卡功能，且已经在无显卡和有显卡的环境下都通过了测试。

|  Net  |         Dataset          |        Parameters        |   CPU time   |   GPU time   |
| :---: | :----------------------: | :----------------------: | :----------: | :----------: |
|  FC   |     Digits (1970×64)     | batch_size=128, epoch=50 | 30.8s±392ms  | 22.4s±298ms  |
| CNN1d | OlivettiFaces (400×4096) | batch_size=64, epoch=50  | 8.76s±68.7ms | 4.49s±16.3ms |
| CNN2d | OlivettiFaces (400×4096) | batch_size=64, epoch=50  | 14.1s±285ms  |  4.54s±49ms  |

事实上，对于越庞大的网络（更宽，更深，卷积），GPU加速效果更好。
