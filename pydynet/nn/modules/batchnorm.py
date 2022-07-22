from .module import Module
from ..parameter import Parameter
from ... import tensor


class BatchNorm1d(Module):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = tensor.zeros(self.num_features)
        self.running_var = tensor.zeros(self.num_features)
        self.scale = Parameter(tensor.randn(self.num_features))
        self.shift = Parameter(tensor.zeros(self.num_features))

    def forward(self, x: tensor):
        if self._train:
            mean = x.mean(0)
            center_data = x - mean
            var = tensor.mean(tensor.square(center_data), 0)
            std_data = center_data / tensor.sqrt(var + self.eps)

            self.running_mean *= (1 - self.momentum)
            self.running_mean += self.momentum * mean
            self.running_var *= (1 - self.momentum)
            self.running_var += self.momentum * var

            return std_data * self.scale + self.shift
        else:
            return (x - self.running_mean) * self.scale / tensor.sqrt(
                self.running_var + self.eps) + self.shift

    def __repr__(self) -> str:
        return "{}(num_features={}, momentum={})".format(
            self.__class__.__name__,
            self.num_features,
            self.momentum,
        )


class BatchNorm2d(Module):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = tensor.zeros((1, self.num_features, 1, 1))
        self.running_var = tensor.zeros((1, self.num_features, 1, 1))
        self.scale = Parameter(tensor.randn(1, self.num_features, 1, 1))
        self.shift = Parameter(tensor.zeros((1, self.num_features, 1, 1)))

    def forward(self, x: tensor):
        if self._train:
            mean = x.mean((0, 2, 3), keepdims=True)
            center_data = x - mean
            var = tensor.mean(tensor.square(center_data), (0, 2, 3),
                              keepdims=True)
            std_data = center_data / tensor.sqrt(var + self.eps)

            self.running_mean *= (1 - self.momentum)
            self.running_mean += self.momentum * mean
            self.running_var *= (1 - self.momentum)
            self.running_var += self.momentum * var

            return std_data * self.scale + self.shift
        else:
            return (x - self.running_mean) * self.scale / tensor.sqrt(
                self.running_var + self.eps) + self.shift

    def __repr__(self) -> str:
        return "{}(num_features={}, momentum={})".format(
            self.__class__.__name__,
            self.num_features,
            self.momentum,
        )