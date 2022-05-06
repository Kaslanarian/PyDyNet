from tensor import Tensor, Graph, zeros, randn, uniform
import functional as F
import numpy as np


class Layer:
    def forward(self, x: Tensor) -> Tensor:
        return

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        return []


class Linear(Layer):
    def __init__(
        self,
        n_input,
        n_output,
    ) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        scale = 1 / self.n_input**0.5
        self.weight = uniform(
            -scale,
            scale,
            (self.n_input, self.n_output),
            requires_grad=True,
        )
        self.bias = uniform(-scale, scale, self.n_output, requires_grad=True)

    def forward(self, x):
        return x @ self.weight + self.bias

    def parameters(self):
        return [self.weight, self.bias]


class Softmax(Layer):
    def forward(self, x):
        return F.softmax(x, axis=-1)


class Dropout(Layer):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        assert p >= 0 and p < 1
        self.p = p
        self.train = True

    def forward(self, x) -> Tensor:
        if self.train:
            return x * Tensor(np.random.binomial(1, 1 - self.p, x.shape[-1]))
        return x * (1 - self.p)


class BatchNorm(Layer):
    def __init__(self, input_size: int, gamma: float = 0.99) -> None:
        super().__init__()
        self.input_size = input_size
        self.gamma = gamma
        self.train = True
        self.running_mean = zeros(self.input_size)
        self.running_var = zeros(self.input_size)
        self.scale = randn(self.input_size, requires_grad=True)
        self.shift = zeros(self.input_size, requires_grad=True)

    def forward(self, x):
        if self.train:
            mean = F.mean(x, 0)
            center_data = x - mean
            var = F.mean(F.square(center_data), 0)
            std_data = center_data / F.sqrt(var)

            self.running_mean.data *= self.gamma
            self.running_mean.data += (1 - self.gamma) * mean.data
            self.running_var.data *= self.gamma
            self.running_var.data += (1 - self.gamma) * var.data

            return std_data * self.scale + self.shift
        else:
            return (x - self.running_mean) * self.scale / F.sqrt(
                self.running_var) + self.shift

    def parameters(self):
        return [self.scale, self.shift]


class Conv1d(Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        scale = 1 / (self.in_channels * self.kernel_size)**0.5
        self.kernel = uniform(
            -scale,
            scale,
            (self.out_channels, self.in_channels, self.kernel_size),
            requires_grad=True,
        )
        self.bias = uniform(
            -scale,
            scale,
            self.out_channels,
            requires_grad=True,
        )

    def forward(self, x):
        return F.conv1d(x, self.kernel, self.padding, self.stride) + self.bias

    def parameters(self):
        return [self.kernel, self.bias]


class Conv2d(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        scale = 1 / (self.in_channels * self.kernel_size**2)**0.5
        self.kernel = uniform(
            -scale,
            scale,
            (self.out_channels, self.in_channels, self.kernel_size,
             self.kernel_size),
            requires_grad=True,
        )
        self.bias = uniform(-scale,
                            scale,
                            self.out_channels,
                            requires_grad=True)

    def forward(self, x):
        return F.conv2d(x, self.kernel, self.padding, self.stride) + self.bias

    def parameters(self):
        return [self.kernel, self.bias]


class RNN(Layer):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity='tanh',
        batch_first=False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.fn = {
            "tanh": F.tanh,
            "relu": F.relu,
        }[nonlinearity]
        scale = 1 / self.hidden_size**0.5
        self.Wxs = [
            uniform(
                -scale,
                scale,
                (self.input_size, self.hidden_size),
                requires_grad=True,
            )
        ] + [
            uniform(
                -scale,
                scale,
                (self.hidden_size, self.hidden_size),
                requires_grad=True,
            ) for i in range(self.num_layers - 1)
        ]
        self.Whs = [
            uniform(
                -scale,
                scale,
                (self.hidden_size, self.hidden_size),
                requires_grad=True,
            ) for i in range(self.num_layers)
        ]
        self.biases = [
            uniform(-scale, scale, self.hidden_size, requires_grad=True)
            for i in range(self.num_layers)
        ]

    def forward_one_layer(self, layer, x, h):
        h_list = []
        if h is None:
            h = zeros(self.hidden_size)

        if self.batch_first:
            for i in range(x.shape[1]):
                h = self.fn(x[..., i:i + 1, :] @ self.Wxs[layer] +
                            h @ self.Whs[layer] + self.biases[layer])
                h_list.append(h)
            return F.concatenate(*h_list, axis=1)
        else:
            for i in range(x.shape[0]):
                h = self.fn(x[i:i + 1] @ self.Wxs[layer] +
                            h @ self.Whs[layer] + self.biases[layer])
                h_list.append(h)
            return F.concatenate(*h_list)

    def forward(self, x: Tensor, h=None):
        '''
        if batch_first:
            x.shape : (batch, seq_len, input_size)
            h.shape : (batch, seq_len, hidden_size)
        else:
            x.shape : (seq_len, batch, input_size)
            h.shape : (seq_len, batch, hidden_size)
        '''
        if h is None:
            h = zeros(self.hidden_size)
        h = self.forward_one_layer(0, x, h)
        for i in range(1, self.num_layers):
            h = self.forward_one_layer(i, h, None)

        return h

    def parameters(self):
        return self.Wxs + self.Whs + self.biases

    def __call__(self, x: Tensor, h=None) -> Tensor:
        return self.forward(x, h)


class LSTM(RNN):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 batch_first=False) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        scale = 1 / self.hidden_size**0.5

        self.Wxs = [
            uniform(-scale,
                    scale, (self.input_size, 4 * self.hidden_size),
                    requires_grad=True)
        ] + [
            uniform(
                -scale,
                scale,
                (self.hidden_size, 4 * self.hidden_size),
                requires_grad=True,
            ) for i in range(self.num_layers - 1)
        ]
        self.Whs = [
            uniform(
                -scale,
                scale,
                (self.hidden_size, 4 * self.hidden_size),
                requires_grad=True,
            ) for i in range(self.num_layers)
        ]
        self.biases = [
            uniform(-scale, scale, 4 * self.hidden_size, requires_grad=True)
            for i in range(self.num_layers)
        ]
        self.c = [zeros(hidden_size) for i in range(self.num_layers)]

    def forward_one_layer(self, layer, x, h):
        h_list = []
        if h is None:
            h = zeros(self.hidden_size)

        if self.batch_first:
            for i in range(x.shape[1]):
                h = self.forward_one_step(layer, x[..., i:i + 1, :], h)
                h_list.append(h)
            return F.concatenate(*h_list, axis=1)
        else:
            for i in range(x.shape[0]):
                h = self.forward_one_step(layer, x[i:i + 1], h)
                h_list.append(h)
            return F.concatenate(*h_list)

    def forward_one_step(self, layer, x: Tensor, h: Tensor):
        affine = x @ self.Wxs[layer] + h @ self.Whs[layer] + self.biases[
            layer]  # 4 * hidden_size
        f_i_o, g = affine[..., :3 *
                          self.hidden_size], affine[..., -self.hidden_size:]
        sigma_fio = F.sigmoid(f_i_o)
        g = F.tanh(g)
        f, i, o = (
            sigma_fio[..., :self.hidden_size],
            sigma_fio[..., self.hidden_size:2 * self.hidden_size],
            sigma_fio[..., 2 * self.hidden_size:],
        )
        self.c[layer] = F.mean(f * self.c[layer] + g * i, (0, 1))
        return o * F.tanh(self.c[layer])

    def forward(self, x: Tensor, h=None):
        if h is None:
            h = zeros(self.hidden_size)

        for i in range(self.num_layers):
            self.c[i] = Tensor(self.c[i].data)

        h = self.forward_one_layer(0, x, h)
        for i in range(1, self.num_layers):
            h = self.forward_one_layer(i, h, None)

        return h

    def parameters(self):
        return self.Wxs + self.Whs + self.biases

    def __call__(self, x: Tensor, h=None) -> Tensor:
        return self.forward(x, h)


class GRU(RNN):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=False,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        scale = 1 / self.hidden_size**0.5
        low_high = (scale, -scale)
        self.Wxs = [
            uniform(
                *low_high,
                (self.input_size, 3 * self.hidden_size),
                requires_grad=True,
            )
        ] + [
            uniform(
                *low_high,
                (self.hidden_size, 3 * self.hidden_size),
                requires_grad=True,
            ) for i in range(self.num_layers - 1)
        ]
        self.Wh_zrs = [
            uniform(
                *low_high,
                (self.hidden_size, 2 * self.hidden_size),
                requires_grad=True,
            ) for i in range(self.num_layers)
        ]
        self.biases_zr = [
            uniform(
                *low_high,
                2 * self.hidden_size,
                requires_grad=True,
            ) for i in range(self.num_layers)
        ]
        self.Whs = [
            uniform(
                *low_high,
                (self.hidden_size, self.hidden_size),
                requires_grad=True,
            ) for i in range(self.num_layers)
        ]
        self.biases = [
            uniform(*low_high, self.hidden_size, requires_grad=True)
            for i in range(self.num_layers)
        ]

    def forward_one_step(self, layer, x: Tensor, h: Tensor):
        affine = x @ self.Wxs[layer]
        zr = F.sigmoid(affine[..., :2 * self.hidden_size] +
                       h @ self.Wh_zrs[layer] + self.biases_zr[layer])
        z, r = zr[..., :self.hidden_size], zr[..., self.hidden_size:]
        h_tilde = F.tanh(affine[..., 2 * self.hidden_size:] +
                         (r * h) @ self.Whs[layer] + self.biases[layer])
        return (1 - z) * h + z * h_tilde

    def forward_one_layer(self, layer, x: Tensor, h=None):
        if h is None:
            h = zeros(self.hidden_size)

        h_list = []
        if self.batch_first:
            for i in range(x.shape[1]):
                h = self.forward_one_step(layer, x[..., i:i + 1, :], h)
                h_list.append(h)
            return F.concatenate(*h_list, axis=1)
        else:
            for i in range(x.shape[0]):
                h = self.forward_one_step(layer, x[i:i + 1], h)
                h_list.append(h)
            return F.concatenate(*h_list)

    def forward(self, x: Tensor, h=None):
        if h is None:
            h = zeros(self.hidden_size)

        h = self.forward_one_layer(0, x, h)
        for i in range(1, self.num_layers):
            h = self.forward_one_layer(i, h, None)

        return h

    def parameters(self):
        return self.Wxs + self.Wh_zrs + self.biases_zr + self.Whs + self.biases

    def __call__(self, x: Tensor, h=None) -> Tensor:
        return self.forward(x, h)


class Module:
    def __init__(self) -> None:
        pass

    def forward(self, x):
        pass

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for member in self.__dict__.values():
            if isinstance(member, Layer):
                params.extend(member.parameters())
        return params

    def train(self):
        for member in self.__dict__.values():
            if type(member) in {Dropout, BatchNorm}:
                member.train = True
        Graph.free_graph()  # 删除测试阶段产生的计算图

    def eval(self):
        for member in self.__dict__.values():
            if type(member) in {Dropout, BatchNorm}:
                member.train = False


class MSELoss:
    def __call__(self, y_pred, y_true):
        return F.mean(F.square(y_pred - y_true))


class NLLLoss:
    def __call__(self, y_pred, y_true):
        return -F.mean(F.log(y_pred) * y_true)


class CrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        update_y_pred = y_pred - np.max(y_pred.data)
        log_sum_exp = F.log(F.sum(F.exp(update_y_pred), 1, keepdims=True))
        return -F.mean((update_y_pred - log_sum_exp) * y_true)
