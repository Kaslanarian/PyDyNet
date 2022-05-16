from collections import OrderedDict
from tensor import Tensor, zeros, randn, uniform
import functional as F
import numpy as np


class Parameter(Tensor):
    def __init__(self, data: Tensor) -> None:
        super().__init__(data.data, True, float)

    def __repr__(self) -> str:
        return "Parameter : {}".format(self.data)


class Module:
    def __init__(self) -> None:
        self._train = True
        self._parameters = OrderedDict()

    def forward(self, x) -> Tensor:
        pass

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
        return tuple(self._parameters.values())

    def train(self):
        self._train = True
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.train()

    def eval(self):
        self._train = False
        for module in self.__dict__.values():
            if isinstance(module, Module):
                module.eval()


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = 1 / self.in_features**0.5
        self.weight = Parameter(
            uniform(
                -scale,
                scale,
                (self.in_features, self.out_features),
            ))
        self.bias = Parameter(uniform(
            -scale,
            scale,
            self.out_features,
        )) if bias else None

    def forward(self, x):
        affine = x @ self.weight
        if self.bias is not None:
            return affine + self.bias
        return affine

    def __repr__(self) -> str:
        return "Linear(in_features={}, out_features={}, bias={})".format(
            self.in_features, self.out_features, self.bias is not None)


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        assert p >= 0 and p < 1
        self.p = p

    def forward(self, x) -> Tensor:
        if self._train:
            return x * Tensor(np.random.binomial(1, 1 - self.p, x.shape[-1]))
        return x * (1 - self.p)

    def __repr__(self) -> str:
        return "{}(p={})".format(self.__class__.__name__, self.p)


class BatchNorm(Module):
    def __init__(self, input_size: int, gamma: float = 0.99) -> None:
        super().__init__()
        self.input_size = input_size
        self.gamma = gamma
        self.running_mean = zeros(self.input_size)
        self.running_var = zeros(self.input_size)
        self.scale = Parameter(randn(self.input_size))
        self.shift = Parameter(zeros(self.input_size))

    def forward(self, x):
        if self._train:
            mean = F.mean(x, 0)
            center_data = x - mean
            var = F.mean(F.square(center_data), 0)
            std_data = center_data / F.sqrt(var)

            self.running_mean *= self.gamma
            self.running_mean += (1 - self.gamma) * mean
            self.running_var *= self.gamma
            self.running_var += (1 - self.gamma) * var

            return std_data * self.scale + self.shift
        else:
            return (x - self.running_mean) * self.scale / F.sqrt(
                self.running_var) + self.shift

    def __repr__(self) -> str:
        return "{}(input_size={}, gamma={})".format(
            self.__class__.__name__,
            self.input_size,
            self.gamma,
        )


class Conv1d(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 bias=True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        scale = 1 / (self.in_channels * self.kernel_size)**0.5
        self.kernel = Parameter(
            uniform(
                -scale,
                scale,
                (self.out_channels, self.in_channels, self.kernel_size),
            ))
        self.bias = Parameter(uniform(
            -scale,
            scale,
            self.out_channels,
        )) if bias else None

    def forward(self, x):
        conv1d = F.conv1d(x, self.kernel, self.padding, self.stride)
        if self.bias is not None:
            return conv1d + self.bias
        return conv1d

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, kernel_size={}, padding={}, stride={}, bias={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.bias is not None,
        )


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        bias=True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        scale = 1 / (self.in_channels * self.kernel_size**2)**0.5
        self.kernel = Parameter(
            uniform(
                -scale,
                scale,
                (self.out_channels, self.in_channels, self.kernel_size,
                 self.kernel_size),
            ))
        self.bias = Parameter(uniform(
            -scale,
            scale,
            self.out_channels,
        )) if bias else None

    def forward(self, x):
        conv2d = F.conv2d(x, self.kernel, self.padding, self.stride)
        if self.bias is not None:
            return conv2d + self.bias
        return conv2d

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, kernel_size={}, padding={}, stride={}, bias={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
            self.stride,
            self.bias is not None,
        )


class MaxPool1d(Module):
    def __init__(self, kernel_size, stride, padding) -> None:
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


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride, padding) -> None:
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


class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding = padding
        self.weight = Parameter(randn(num_embeddings, embedding_dim))
        if padding:
            self.padding_weight = F.concatenate(
                zeros(embedding_dim),
                self.weight,
            )
        else:
            self.padding_weight = self.weight

    def forward(self, x) -> Tensor:
        return self.padding_weight[x]

    def __repr__(self) -> str:
        return "{}(num_embeddings={}, embedding_dim={}, padding={})".format(
            self.__class__.__name__,
            self.num_embeddings,
            self.embedding_dim,
            self.padding,
        )


# 激活函数
class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)

    def __repr__(self) -> str:
        return "{}()".format(self.__class__.__name__)


class Tanh(Sigmoid):
    def forward(self, x):
        return F.tanh(x)


class ReLU(Sigmoid):
    def forward(self, x):
        return F.relu(x)


class LeakyReLU(Module):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return F.leaky_relu(x, self.alpha)

    def __repr__(self) -> str:
        return "{}(alpha={})".format(self.__class__.__name__, self.alpha)


# 损失函数
class MSELoss(Module):
    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
        assert self.reduction in {'mean', 'sum'}

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.mse_loss(y_pred, y_true, reduction=self.reduction)


class NLLLoss(MSELoss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.nll_loss(y_pred, y_true, reduction=self.reduction)


class CrossEntropyLoss(MSELoss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.cross_entropy_loss(y_pred, y_true, reduction=self.reduction)


# RNN
class RNN(Module):
    '''
    单层RNN，不像Pytorch那样可以指定num_layers进行多层堆叠，我们的RNN是单层可双向的，
    如果要搭建多层RNN:

    ```python
    class MultiLayerRNN(Module):
        def __init__(self):
            super().__init__()
            self.rnn1 = RNN(...)
            self.rnn2 = RNN(...)
            ...

        def forward(x, h=None):
            x = self.rnn1(x, h)
            return self.rnn2(x)
    ```
    '''
    def __init__(
        self,
        input_size,
        hidden_size,
        nonlinearity='tanh',
        batch_first=False,
        bidirectional=False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity
        self.fn = {
            "tanh": F.tanh,
            "relu": F.relu,
        }[nonlinearity]
        scale = 1 / self.hidden_size**0.5
        low_high = (-scale, scale)

        self.Wx = Parameter(uniform(*low_high, (input_size, hidden_size)))
        self.Wh = Parameter(uniform(*low_high, (hidden_size, hidden_size)))
        self.bias = Parameter(uniform(*low_high, hidden_size))

        if self.bidirectional:
            self.Wx_reverse = Parameter(
                uniform(*low_high, (input_size, hidden_size)))
            self.Wh_reverse = Parameter(
                uniform(*low_high, (hidden_size, hidden_size)))
            self.bias_reverse = Parameter(uniform(*low_high, hidden_size))

    def forward(self, x: Tensor, h=(None, None)):
        '''
        if batch_first:
            x.shape : (batch, seq_len, input_size)
            h.shape : (batch, seq_len, hidden_size)
        else:
            x.shape : (seq_len, batch, input_size)
            h.shape : (seq_len, batch, hidden_size)
        '''
        h, h_reverse = h
        if h is None:
            h = zeros(self.hidden_size)
        if h_reverse is None:
            h_reverse = zeros(self.hidden_size)

        if self.batch_first and x.ndim == 3:
            # x.ndim的数据是(seq_len, input_size)不需要变换
            x = x.transpose(1, 0, 2)

        h_list = []
        if self.bidirectional:
            h_reverse_list = []
            for i in range(x.shape[0]):
                h = self.fn(x[i:i + 1] @ self.Wx + h @ self.Wh + self.bias)
                h_list.append(h)
                h_reverse = self.fn(
                    x[x.shape[0] - i - 1:x.shape[0] -
                      i]) @ self.Wx + h_reverse @ self.Wh + self.bias_reverse
                h_reverse_list.append(h_reverse)
            output = F.concatenate(
                F.concatenate(*h_list),
                F.concatenate(*h_reverse_list),
                axis=-1,
            )
        else:
            for i in range(x.shape[0]):
                h = self.fn(x[i:i + 1] @ self.Wx + h @ self.Wh + self.bias)
                h_list.append(h)
            output = F.concatenate(*h_list)

        if self.batch_first and x.ndim == 3:
            output = output.transpose(1, 0, 2)
        return output

    def __repr__(self) -> str:
        return "{}(input_size={}, hidden_size={}, nonlinearity={}, batch_first={}, bidirectional={})".format(
            self.__class__.__name__, self.input_size, self.hidden_size,
            self.nonlinearity, self.batch_first, self.bidirectional)


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=False,
        bidirectional=False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        scale = 1 / self.hidden_size**0.5
        low_high = (-scale, scale)

        self.Wx = Parameter(uniform(*low_high, (input_size, hidden_size * 4)))
        self.Wh = Parameter(uniform(*low_high, (hidden_size, hidden_size * 4)))
        self.bias = Parameter(uniform(*low_high, hidden_size * 4))

        if self.bidirectional:
            self.Wx_reverse = Parameter(
                uniform(*low_high, (input_size, hidden_size * 4)))
            self.Wh_reverse = Parameter(
                uniform(*low_high, (hidden_size, hidden_size * 4)))
            self.bias_reverse = Parameter(uniform(*low_high, hidden_size * 4))

    def forward(self, x: Tensor, h=(None, None)) -> Tensor:
        c = zeros(self.hidden_size)
        c_reverse = zeros(self.hidden_size)

        h, h_reverse = h
        if h is None:
            h = zeros(self.hidden_size)
        if h_reverse is None:
            h_reverse = zeros(self.hidden_size)

        if self.batch_first and x.ndim == 3:
            x = x.transpose(1, 0, 2)

        h_list = []

        if self.bidirectional:
            h_reverse_list = []
            for id in range(x.shape[0]):
                affine = x[id:id + 1] @ self.Wx + h @ self.Wh + self.bias
                f_i_o = affine[..., :3 * self.hidden_size]
                g = affine[..., -self.hidden_size:]
                sigma_fio = F.sigmoid(f_i_o)
                g = F.tanh(g)
                f = sigma_fio[..., :self.hidden_size]
                i = sigma_fio[..., self.hidden_size:2 * self.hidden_size]
                o = sigma_fio[..., -self.hidden_size:]
                c = F.mean(f * c + g * i, (0, 1))
                h = o * F.tanh(c)
                h_list.append(h)

                affine = x[
                    x.shape[0] - id - 1:x.shape[0] -
                    id] @ self.Wx_reverse + h_reverse @ self.Wh_reverse + self.bias_reverse
                f_i_o = affine[..., :3 * self.hidden_size]
                g = affine[..., -self.hidden_size:]
                sigma_fio = F.sigmoid(f_i_o)
                g = F.tanh(g)
                f = sigma_fio[..., :self.hidden_size]
                i = sigma_fio[..., self.hidden_size:2 * self.hidden_size]
                o = sigma_fio[..., -self.hidden_size:]
                c = F.mean(f * c_reverse + g * i, (0, 1))
                h_reverse = o * F.tanh(c_reverse)
                h_reverse_list.append(h_reverse)

                output = F.concatenate(
                    F.concatenate(*h_list),
                    F.concatenate(*h_reverse_list),
                    axis=-1,
                )
        else:
            for id in range(x.shape[0]):
                affine = x[id:id + 1] @ self.Wx + h @ self.Wh + self.bias
                f_i_o = affine[..., :3 * self.hidden_size]
                g = affine[..., -self.hidden_size:]
                sigma_fio = F.sigmoid(f_i_o)
                g = F.tanh(g)
                f = sigma_fio[..., :self.hidden_size]
                i = sigma_fio[..., self.hidden_size:2 * self.hidden_size]
                o = sigma_fio[..., -self.hidden_size:]
                c = F.mean(f * c + g * i, (0, 1))
                h = o * F.tanh(c)
                h_list.append(h)
            output = F.concatenate(*h_list)

        if self.batch_first and x.ndim == 3:
            output = output.transpose(1, 0, 2)
        return output

    def __repr__(self) -> str:
        return "{}(input_size={}, hidden_size={}, batch_first={}, bidirectional={})".format(
            self.__class__.__name__, self.input_size, self.hidden_size,
            self.batch_first, self.bidirectional)


class GRU(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=False,
        bidirectional=False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        scale = 1 / self.hidden_size**0.5
        low_high = (scale, -scale)

        self.Wx = Parameter(uniform(*low_high, (input_size, 3 * hidden_size)))
        self.Wh_zr = Parameter(
            uniform(*low_high, (hidden_size, 2 * hidden_size)))
        self.bias_zr = Parameter(uniform(*low_high, 2 * hidden_size))
        self.Wh = Parameter(uniform(*low_high, (hidden_size, hidden_size)))
        self.bias = Parameter(uniform(*low_high, self.hidden_size))

        if self.bidirectional:
            self.Wx_reverse = Parameter(
                uniform(*low_high, (input_size, 3 * hidden_size)))
            self.Wh_zr_reverse = Parameter(
                uniform(*low_high, (hidden_size, 2 * hidden_size)))
            self.bias_zr_recerse = Parameter(
                uniform(*low_high, 2 * hidden_size))
            self.Wh_reverse = Parameter(
                uniform(*low_high, (hidden_size, hidden_size)))
            self.bias_reverse = Parameter(uniform(*low_high, self.hidden_size))

    def forward(self, x: Tensor, h=(None, None)) -> Tensor:
        h, h_reverse = h
        if h is None:
            h = zeros(self.hidden_size)
        if h_reverse is None:
            h_reverse = zeros(self.hidden_size)

        if self.batch_first and x.ndim == 3:
            # x.ndim的数据是(seq_len, input_size)不需要变换
            x = x.transpose(1, 0, 2)

        h_list = []

        if self.bidirectional:
            h_reverse_list = []
            for i in range(x.shape[0]):
                affine = x[i:i + 1] @ self.Wx
                zr = F.sigmoid(affine[..., :2 * self.hidden_size] +
                               h @ self.Wh_zr + self.bias_zr)
                z, r = zr[..., :self.hidden_size], zr[..., self.hidden_size:]
                h_tilde = F.tanh(affine[..., -self.hidden_size:] +
                                 (r * h) @ self.Wh + self.bias)
                h = (1 - z) * h + z * h_tilde
                h_list.append(h)

                affine = x[x.shape[0] - i - 1:x.shape[0] - i] @ self.Wx_reverse
                zr = F.sigmoid(affine[..., :2 * self.hidden_size] +
                               h @ self.Wh_zr_reverse + self.bias_zr_recerse)
                z, r = zr[..., :self.hidden_size], zr[..., self.hidden_size:]
                h_tilde = F.tanh(affine[..., -self.hidden_size:] +
                                 (r * h) @ self.Wh_reverse + self.bias_reverse)
                h_reverse = (1 - z) * h_reverse + z * h_tilde
                h_reverse_list.append(h_reverse)

                output = F.concatenate(
                    F.concatenate(*h_list),
                    F.concatenate(*h_reverse_list),
                    axis=-1,
                )

        else:
            for i in range(x.shape[0]):
                affine = x[i:i + 1] @ self.Wx
                zr = F.sigmoid(affine[..., :2 * self.hidden_size] +
                               h @ self.Wh_zr + self.bias_zr)
                z, r = zr[..., :self.hidden_size], zr[..., self.hidden_size:]
                h_tilde = F.tanh(affine[..., -self.hidden_size:] +
                                 (r * h) @ self.Wh + self.bias)
                h = (1 - z) * h + z * h_tilde
                h_list.append(h)
            output = F.concatenate(*h_list)

        if self.batch_first and x.ndim == 3:
            output = output.transpose(1, 0, 2)
        return output

    def __repr__(self) -> str:
        return "{}(input_size={}, hidden_size={}, batch_first={}, bidirectional={})".format(
            self.__class__.__name__, self.input_size, self.hidden_size,
            self.batch_first, self.bidirectional)