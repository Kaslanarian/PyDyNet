import numpy as np


class SGD:
    def __init__(self, params, lr=0.1, weight_decay=0.) -> None:
        self.lr = lr
        self.params = params
        self.weight_devay = weight_decay

    def step(self):
        for param in self.params:
            param.data -= self.lr * (param.grad +
                                     self.weight_devay * param.data)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()


class Momentum(SGD):
    def __init__(self, params, lr=0.1, momentum=0.5, weight_decay=0.) -> None:
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum
        self.v = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            self.v[i] *= self.momentum
            self.v[i] += self.lr * (self.params[i].grad +
                                    self.weight_devay * self.params[i].data)
            self.params[i].data -= self.v[i]


class Adagrad(SGD):
    def __init__(self, params, lr=0.01, weight_decay=0.) -> None:
        super().__init__(params, lr, weight_decay)
        self.G = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_devay * self.params[i].data
            self.G[i] += grad**2
            self.params[i].data -= self.lr * grad / np.sqrt(1e-8 + self.G[i])


class Adadelta(Adagrad):
    def __init__(self, params, lr=0.001, gamma=0.9, weight_decay=0.) -> None:
        super().__init__(params, lr, weight_decay)
        self.gamma = gamma

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_devay * self.params[i].data
            self.G[i] = self.gamma * self.G[i] + (1 - self.gamma) * grad**2
            self.params[i].data -= self.lr * grad / np.sqrt(self.G[i] + 1e-8)


class Adam(SGD):
    def __init__(self,
                 params,
                 lr=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 weight_decay=0.) -> None:
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
