from typing import Any
import numpy as np

try:
    import cupy as cp
    cuda_available: bool = True
except ImportError as e:
    cp = object()
    cuda_available: bool = False
    print(e)


def is_available() -> bool:
    return cuda_available


def device_count() -> int:
    if is_available():
        return cp.cuda.runtime.getDeviceCount()
    else:
        return 0


def current_device() -> int:
    return cp.cuda.runtime.getDevice()


def set_device(device: int) -> None:
    return cp.cuda.runtime.setDevice(device)


class Device:
    def __init__(self, device: Any = None) -> None:
        if isinstance(device, str):
            if device == "cpu":
                self.device = "cpu"
            elif device == "cuda":
                self.device = "cuda"
                self.device_id = current_device()
            else:
                assert len(device) > 5 and device[:5] == "cuda:" and device[
                    5:].isdigit()
                self.device = "cuda"
                self.device_id = int(device[5:])
        elif isinstance(device, int):
            self.device = "cuda"
            self.device_id = device
        elif device is None:
            self.device = "cpu"
        elif isinstance(device, Device):
            self.device = device.device
            if self.device != "cpu":
                self.device_id = device.device_id
        if self.device == "cuda":
            self.device = cp.cuda.Device(self.device_id)
        assert self.device == "cpu" or is_available()

    def __repr__(self) -> str:
        if self.device == "cpu":
            return "Device(type='cpu')"
        else:
            return "Device(type='cuda', index={})".format(self.device_id)

    def __eq__(self, device: Any) -> bool:
        assert isinstance(device, Device)
        if self.device == "cpu":
            return device.device == "cpu"
        else:
            if device.device == "cpu":
                return False
            return self.device == device.device

    @property
    def xp(self):
        return np if self.device == "cpu" else cp

    def __enter__(self):
        if self.device != "cpu":
            return self.device.__enter__()

    def __exit__(self, type, value, trace):
        if self.device != "cpu":
            return self.device.__exit__(type, value, trace)
