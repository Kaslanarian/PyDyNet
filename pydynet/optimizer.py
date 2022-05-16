'''优化器类，我们目前实现了\n
- SGD;\n
- Momentum;\n
- Adagrad;\n
- Adadelta;\n
- Adam.\n

Reference
---------
论文: https://arxiv.org/abs/1609.04747;\n
博客: https://welts.xyz/2021/08/20/gd/.
'''
import numpy as np


class SGD:
    '''批梯度下降优化器:

    .. math:: \\theta=\\theta-\\eta\\cdot\\nabla_{\\theta}J(\\theta;x^{(i:i+n)}, y^{(i:i+n)})

    Parameters
    ----------
    params : tuple
        待优化参数元组;
    lr : float
        学习率;
    weight_decay : float, default=0.
        权重衰减系数.
    '''
    def __init__(self,
                 params: tuple,
                 lr: float,
                 weight_decay: float = 0.) -> None:
        self.lr = lr
        self.params = params
        self.weight_devay = weight_decay

    def step(self):
        '''对所有参数进行上式的更新.'''
        for param in self.params:
            param.data -= self.lr * (param.grad +
                                     self.weight_devay * param.data)

    def zero_grad(self):
        '''针对self.params梯度清零.'''
        for param in self.params:
            param.zero_grad()


class Momentum(SGD):
    '''带动量的梯度下降

    .. math::
        v_t &= \\gamma v_{t-1}+\\eta\\nabla_{\\theta}J(\\theta) \\\\
        \\theta &= \\theta-v_t

    Parameters
    ----------
    params : tuple
        待优化参数元组;
    lr : float
        学习率;
    momentum : float, default=0.5
        动量系数;
    weight_decay : float, default=0.
        权重衰减系数.
    '''
    def __init__(self,
                 params: tuple,
                 lr: float,
                 momentum: float = 0.5,
                 weight_decay: float = 0.) -> None:
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            self.v[i] *= self.momentum
            self.v[i] += self.lr * (self.params[i].grad +
                                    self.weight_devay * self.params[i].data)
            self.params[i].data -= self.v[i]

    def zero_grad(self):
        return super().zero_grad()


class Adagrad(SGD):
    '''Adagrad下降法

    Parameters
    ----------
    params : tuple
        待优化参数元组;
    lr : float
        学习率;
    weight_decay : float, default=0.
        权重衰减系数.
    '''
    def __init__(self, params, lr=0.01, weight_decay=0.) -> None:
        super().__init__(params, lr, weight_decay)
        self.G = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_devay * self.params[i].data
            self.G[i] += grad**2
            self.params[i].data -= self.lr * grad / np.sqrt(1e-8 + self.G[i])

    def zero_grad(self):
        return super().zero_grad()


class Adadelta(Adagrad):
    '''Adadelta下降法

    Parameters
    ----------
    params : tuple
        待优化参数元组;
    lr : float
        学习率;
    gamma : float, default=0.9
        滑动系数 :math:`\\gamma`，默认0.9;
    weight_decay : float, default=0.
        权重衰减系数.
    '''
    def __init__(self,
                 params: tuple,
                 lr: float,
                 gamma: float = 0.9,
                 weight_decay: float = 0.) -> None:
        super().__init__(params, lr, weight_decay)
        self.gamma = gamma

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_devay * self.params[i].data
            self.G[i] = self.gamma * self.G[i] + (1 - self.gamma) * grad**2
            self.params[i].data -= self.lr * grad / np.sqrt(self.G[i] + 1e-8)

    def zero_grad(self):
        return super().zero_grad()


class Adam(SGD):
    '''Adam下降法
    
    Parameters
    ----------
    params : tuple
        待优化参数元组;
    lr : float
        学习率;
    beta1 : float, default=0.9
        Adam参数1，默认0.9;
    beta2 : float, default=0.999
        Adam参数2，默认0.999;
    weight_decay : float, default=0.
        权重衰减系数.
    '''
    def __init__(self,
                 params: tuple,
                 lr: float,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 weight_decay: float = 0.) -> None:
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = beta1, beta2
        self.m = [np.zeros(param.shape) for param in self.params]
        self.v = [np.zeros(param.shape) for param in self.params]
        self.t = 1

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_devay * self.params[i].data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            m_t = self.m[i] / (1 - self.beta1**self.t)
            v_t = self.v[i] / (1 - self.beta2**self.t)

            self.params[i].data -= self.lr * m_t / (np.sqrt(v_t) + 1e-8)
        self.t += 1

    def zero_grad(self):
        return super().zero_grad()