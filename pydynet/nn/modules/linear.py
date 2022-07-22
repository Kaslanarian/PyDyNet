from .module import Module
from ..parameter import Parameter
from .. import init
from .. import functional as F
from ...tensor import Tensor, empty

import math


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(empty((self.in_features, self.out_features)))
        self.bias = Parameter(empty(self.out_features)) if bias else None
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