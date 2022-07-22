from .module import Module
from ..parameter import Parameter
from .. import init
from .. import functional as F
from ... import tensor

import math


class Conv1d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = Parameter(
            tensor.empty(
                (self.out_channels, self.in_channels, self.kernel_size)))
        self.bias = Parameter(tensor.empty(
            (1, self.out_channels, 1))) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        conv1d = F.conv1d(x, self.weight, self.padding, self.stride)
        if self.bias is not None:
            return conv1d + self.bias
        return conv1d

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, kernel_size={}, padding={}, stride={}, bias={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.bias is not None,
        )


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        bias=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weight = Parameter(
            tensor.empty((self.out_channels, self.in_channels,
                          self.kernel_size, self.kernel_size)))
        self.bias = Parameter(tensor.empty(
            (1, self.out_channels, 1, 1))) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        conv2d = F.conv2d(x, self.weight, self.padding, self.stride)
        if self.bias is not None:
            return conv2d + self.bias
        return conv2d

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, kernel_size={}, padding={}, stride={}, bias={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.bias is not None,
        )


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self) -> str:
        return "{}(kernel_size={}, stride={}, padding={})".format(
            self.__class__.__name__,
            self.kernel_size,
            self.stride,
            self.padding,
        )


class AvgPool1d(Module):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.avg_pool1d(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self) -> str:
        return "{}(kernel_size={}, stride={}, padding={})".format(
            self.__class__.__name__,
            self.kernel_size,
            self.stride,
            self.padding,
        )


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self) -> str:
        return "{}(kernel_size={}, stride={}, padding={})".format(
            self.__class__.__name__,
            self.kernel_size,
            self.stride,
            self.padding,
        )


class AvgPool2d(Module):
    def __init__(self, kernel_size, stride, padding) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)

    def __repr__(self) -> str:
        return "{}(kernel_size={}, stride={}, padding={})".format(
            self.__class__.__name__,
            self.kernel_size,
            self.stride,
            self.padding,
        )