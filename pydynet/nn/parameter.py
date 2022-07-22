from ..tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data, True, float)

    def __repr__(self) -> str:
        return "Parameter : {}".format(self.data)