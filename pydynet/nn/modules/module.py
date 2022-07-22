from collections import OrderedDict
from ..parameter import Parameter
from ...tensor import Tensor


class Module:
    def __init__(self) -> None:
        self._train = True
        self._parameters = OrderedDict()

    def __call__(self, *x) -> Tensor:
        return self.forward(*x)

    def __setattr__(self, __name: str, __value) -> None:
        self.__dict__[__name] = __value
        if isinstance(__value, Parameter):
            self._parameters[__name] = __value
        if isinstance(__value, Module):
            for key in __value._parameters:
                self._parameters[__name + "." + key] = __value._parameters[key]

    def __repr__(self) -> str:
        module_list = [
            module for module in self.__dict__.items()
            if isinstance(module[1], Module)
        ]
        return "{}(\n{}\n)".format(
            self.__class__.__name__,
            "\n".join([
                "{:>10} : {}".format(module_name, module)
                for module_name, module in module_list
            ]),
        )

    def parameters(self):
        for param in self._parameters.values():
            yield param

    def train(self):
        self._train = True
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.train()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def eval(self):
        self._train = False
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.eval()


class Sequential(Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.module_list = []
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for name, module in args[0].items():
                self.__setattr__(name, module)
                self.module_list.append(module)
        else:
            for idx, module in enumerate(args):
                self.__setattr__(str(idx), module)
                self.module_list.append(module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self.module_list:
            x = module(x)
        return x

    def __len__(self):
        return len(self.module_list)