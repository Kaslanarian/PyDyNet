from .module import Module
from .. import functional as F
from ...tensor import Tensor


class Sigmoid(Module):
    def forward(self, x) -> Tensor:
        return F.sigmoid(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class Tanh(Module):
    def forward(self, x) -> Tensor:
        return F.tanh(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class ReLU(Module):
    def forward(self, x) -> Tensor:
        return F.relu(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class LeakyReLU(Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x) -> Tensor:
        return F.leaky_relu(x, self.alpha)

    def __repr__(self) -> str:
        return "{}(alpha={})".format(self.__class__.__name__, self.alpha)


class Softmax(Module):
    def __init__(self, axis=None) -> None:
        super().__init__()
        self.axis = axis

    def forward(self, x) -> Tensor:
        return F.softmax(x, self.axis)

    def __repr__(self) -> str:
        return "{}(axis={})".format(self.__class__.__name__, self.axis)