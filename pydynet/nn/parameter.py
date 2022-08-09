from ..tensor import Tensor
from ..cuda import Device


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(
            data=data.data,
            dtype=data.dtype,
            device=data.device,
            requires_grad=True,
        )

    def __repr__(self) -> str:
        return "Parameter : \n{}".format(self.data) + (",\ndevice={}".format(
            self.device) if self.device.device != "cpu" else "")

    def to(self, device):
        device = Device(device)
        if self.device == device:
            return self
        elif device.device == "cpu":  # cuda -> cpu
            return self.__class__(
                Tensor(
                    self.data.get(),
                    dtype=self.dtype,
                    device=device,
                ))
        else:  # cpu -> cuda
            return self.__class__(
                Tensor(
                    self.data,
                    dtype=self.dtype,
                    device=device,
                ))
