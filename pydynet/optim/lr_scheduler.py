'''学习率调节器类，我们目前实现了\n
- ExponentialLR;\n
- StepLR;\n
- MultiStepLR;\n
- CosineAnnealingLR.\n
'''

from typing import List
from .optimizer import Optimizer
import weakref
from functools import wraps
from collections import Counter
from math import cos, pi


class _LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch

        if self.last_epoch == -1:
            self.optimizer.initial_lr = self.optimizer.lr
        else:
            assert hasattr(
                self.optimizer, "initial_lr"
            ), "last_epoch=1 but no 'initial_lr' attribute in optimizer!"

        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # 建立一个method的弱引用。弱引用不增加对象的引用计数,只存在弱引用的对象是可被垃圾回收的;
            # 弱引用可以解决循环引用的问题。
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__  # __func__是method的底层实现,不跟具体的实例绑定
            cls = instance_ref().__class__  # method的所属类
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        # 通过装饰器来为optimizer.step添加计数功能,并初始化计数器
        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0

        self.step()

    def step(self):
        self._step_count += 1  # lr_scheduler的step计数

        # 支持上下文管理器协议的类
        class _enable_get_lr_call:
            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            self.last_epoch += 1  # 更新epoch
            lr = self.get_lr()  # 计算新的lr,与具体的lr_scheduler类型有关

        # _last_lr记录上一轮次更新的lr值
        self._last_lr = self.optimizer.lr
        self.optimizer.lr = lr

    def get_lr(self):
        raise NotImplementedError

    def get_last_lr(self):
        return self._last_lr


class ExponentialLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.optimizer.lr * self.gamma**self.last_epoch


class StepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma=0.1,
        last_epoch: int = -1,
    ) -> None:
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.optimizer.lr * self.gamma**(self.last_epoch //
                                                self.step_size)


class MultiStepLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma=0.1,
        last_epoch: int = -1,
    ) -> None:
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return self.optimizer.lr
        return self.optimizer.lr * self.gamma**self.milestones[self.last_epoch]


class CosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        base_lr = self.optimizer.initial_lr
        if self.last_epoch == 0:
            return base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.get_last_lr() + (base_lr - self.eta_min) * (
                1 - cos(pi / self.T_max)) / 2
        return (1 + cos(pi * self.last_epoch / self.T_max)) / (
            1 + cos(pi * (self.last_epoch - 1) / self.T_max)) * (
                self.get_last_lr() - self.eta_min) + self.eta_min
