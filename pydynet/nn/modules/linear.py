from ctypes import Union
from .module import Module
from ..parameter import Parameter
from .. import init
from .. import functional as F
from ...tensor import Tensor, empty
from ...cuda import Device

import math


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        kwargs = {"device": Device(device), "dtype": dtype}
        self.weight = Parameter(
            empty((self.in_features, self.out_features), **kwargs))
        self.bias = Parameter(empty(self.out_features, **
                                    kwargs)) if bias else None
        self.reset_paramters()

    def reset_paramters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return "Linear(in_features={}, out_features={}, bias={})".format(
            self.in_features, self.out_features, self.bias is not None)

    def move(self, device):
        self.device = device
        self.weight = self.weight.to(self.device)
        self.bias = self.bias.to(self.device)