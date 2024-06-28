from .module import Module
from ..parameter import Parameter
from ...special import zeros, randn
from ... import tensor
from ...cuda import Device


class BatchNorm1d(Module):
    '''
    一维Batch Normalization层

    Parameters
    ----------
    num_features : int
        输入特征数.
    eps : float, default=1e-5
        防止除数为0的极小项.
    momentum : float, default=0.5
        计算累积均值和方差的动量项.
    device : Optional[Device], default=None
        层数据所在的设备.
    dtype : default=Nonr
        层数据的类型.
    '''

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": Device(device), "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = Parameter(
            zeros(self.num_features, **kwargs),
            requires_grad=False,
        )
        self.running_var = Parameter(
            zeros(self.num_features, **kwargs),
            requires_grad=False,
        )
        self.scale = Parameter(randn(self.num_features, **kwargs))
        self.shift = Parameter(zeros(self.num_features, **kwargs))

    def forward(self, x):
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
    '''
    二维Batch Normalization层

    Parameters
    ----------
    num_features : int
        输入特征数(通道数).
    eps : float, default=1e-5
        防止除数为0的极小项.
    momentum : float, default=0.5
        计算累积均值和方差的动量项.
    device : Optional[Device], default=None
        层数据所在的设备.
    dtype : default=None
        层数据的类型.
    '''

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": Device(device), "dtype": dtype}
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = Parameter(
            zeros((1, self.num_features, 1, 1), **kwargs),
            requires_grad=False,
        )
        self.running_var = Parameter(
            zeros((1, self.num_features, 1, 1), **kwargs),
            requires_grad=False,
        )
        self.scale = Parameter(randn(1, self.num_features, 1, 1, **kwargs))
        self.shift = Parameter(zeros((1, self.num_features, 1, 1), **kwargs))

    def forward(self, x):
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


class LayerNorm(Module):
    '''
    Layer Normalization

    Parameters
    ----------
    normalized_shape : Tuple[int]
    eps : float, default=1e-5
    momentum : float, default=0.5
    device : Optional[Device], default=None
    dtype : default=None
    '''

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        kwargs = {"device": Device(device), "dtype": dtype}
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.momentum = momentum
        self.running_mean = Parameter(
            zeros(normalized_shape, **kwargs),
            requires_grad=False,
        )
        self.running_var = Parameter(
            zeros(normalized_shape, **kwargs),
            requires_grad=False,
        )
        self.scale = Parameter(randn(*normalized_shape, **kwargs))
        self.shift = Parameter(zeros(normalized_shape, **kwargs))

    def forward(self, x):
        if self._train:
            mean = x.mean(*range(x.ndim - len(self.normalized_shape)))
            center_data = x - mean
            var = tensor.mean(
                tensor.square(center_data),
                tuple(range(x.ndim - len(self.normalized_shape))),
            )
            std_data = center_data / tensor.sqrt(var + self.eps)
            self.running_mean *= (1 - self.momentum)
            self.running_mean += self.momentum * mean
            self.running_var *= (1 - self.momentum)
            self.running_var += self.momentum * var

            return std_data * self.scale + self.shift
        else:
            return (x - self.running_mean) * self.scale / tensor.sqrt(
                self.running_var + self.eps) + self.shift
