from .module import Module
from ...tensor import Tensor


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        assert p >= 0 and p < 1
        self.p = p

    def forward(self, x) -> Tensor:
        if self._train:
            return x * x.xp.random.binomial(1, 1 - self.p, x.shape[-1])
        return x * (1 - self.p)

    def __repr__(self) -> str:
        return "{}(p={})".format(self.__class__.__name__, self.p)