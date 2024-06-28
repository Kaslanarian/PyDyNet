from .module import Module
from .. import functional as F


class MaxPool1d(Module):

    def __init__(self, kernel_size: int, stride: int, padding: int) -> None:
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

    def __init__(self, kernel_size: int, stride: int, padding: int) -> None:
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

    def __init__(self, kernel_size: int, stride: int, padding: int) -> None:
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

    def __init__(self, kernel_size: int, stride: int, padding: int) -> None:
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
