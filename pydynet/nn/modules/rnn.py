from .module import Module
from .. import functional as F
from ..parameter import Parameter
from ... import tensor


class RNN(Module):
    '''单层RNN，不像Pytorch那样可以指定num_layers进行多层堆叠，我们的RNN是单层可双向的，'''
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

        self.Wx = Parameter(
            tensor.uniform(*low_high, (input_size, hidden_size)))
        self.Wh = Parameter(
            tensor.uniform(*low_high, (hidden_size, hidden_size)))
        self.bias = Parameter(tensor.uniform(*low_high, hidden_size))

        if self.bidirectional:
            self.Wx_reverse = Parameter(
                tensor.uniform(*low_high, (input_size, hidden_size)))
            self.Wh_reverse = Parameter(
                tensor.uniform(*low_high, (hidden_size, hidden_size)))
            self.bias_reverse = Parameter(
                tensor.uniform(*low_high, hidden_size))

    def forward(self, x: tensor.Tensor, h=(None, None)):
        h, h_reverse = h
        if h is None:
            h = tensor.zeros(self.hidden_size)
        if h_reverse is None:
            h_reverse = tensor.zeros(self.hidden_size)

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
            output = tensor.concatenate(
                tensor.concatenate(*h_list),
                tensor.concatenate(*h_reverse_list),
                axis=-1,
            )
        else:
            for i in range(x.shape[0]):
                h = self.fn(x[i:i + 1] @ self.Wx + h @ self.Wh + self.bias)
                h_list.append(h)
            output = tensor.concatenate(*h_list)

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

        self.Wx = Parameter(
            tensor.uniform(*low_high, (input_size, hidden_size * 4)))
        self.Wh = Parameter(
            tensor.uniform(*low_high, (hidden_size, hidden_size * 4)))
        self.bias = Parameter(tensor.uniform(*low_high, hidden_size * 4))

        if self.bidirectional:
            self.Wx_reverse = Parameter(
                tensor.uniform(*low_high, (input_size, hidden_size * 4)))
            self.Wh_reverse = Parameter(
                tensor.uniform(*low_high, (hidden_size, hidden_size * 4)))
            self.bias_reverse = Parameter(
                tensor.uniform(*low_high, hidden_size * 4))

    def forward(self, x: tensor.Tensor, h=(None, None)) -> tensor.Tensor:
        c = tensor.zeros(self.hidden_size)
        c_reverse = tensor.zeros(self.hidden_size)

        h, h_reverse = h
        if h is None:
            h = tensor.zeros(self.hidden_size)
        if h_reverse is None:
            h_reverse = tensor.zeros(self.hidden_size)

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
                c = tensor.sum(f * c + g * i, (0, 1))
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
                c = tensor.sum(f * c_reverse + g * i, (0, 1))
                h_reverse = o * F.tanh(c_reverse)
                h_reverse_list.append(h_reverse)

                output = tensor.concatenate(
                    tensor.concatenate(*h_list),
                    tensor.concatenate(*h_reverse_list),
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
            output = tensor.concatenate(*h_list)

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

        self.Wx = Parameter(
            tensor.uniform(*low_high, (input_size, 3 * hidden_size)))
        self.Wh_zr = Parameter(
            tensor.uniform(*low_high, (hidden_size, 2 * hidden_size)))
        self.bias_zr = Parameter(tensor.uniform(*low_high, 2 * hidden_size))
        self.Wh = Parameter(
            tensor.uniform(*low_high, (hidden_size, hidden_size)))
        self.bias = Parameter(tensor.uniform(*low_high, self.hidden_size))

        if self.bidirectional:
            self.Wx_reverse = Parameter(
                tensor.uniform(*low_high, (input_size, 3 * hidden_size)))
            self.Wh_zr_reverse = Parameter(
                tensor.uniform(*low_high, (hidden_size, 2 * hidden_size)))
            self.bias_zr_recerse = Parameter(
                tensor.uniform(*low_high, 2 * hidden_size))
            self.Wh_reverse = Parameter(
                tensor.uniform(*low_high, (hidden_size, hidden_size)))
            self.bias_reverse = Parameter(
                tensor.uniform(*low_high, self.hidden_size))

    def forward(self, x: tensor.Tensor, h=(None, None)) -> tensor.Tensor:
        h, h_reverse = h
        if h is None:
            h = tensor.zeros(self.hidden_size)
        if h_reverse is None:
            h_reverse = tensor.zeros(self.hidden_size)

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

                output = tensor.concatenate(
                    tensor.concatenate(*h_list),
                    tensor.concatenate(*h_reverse_list),
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
            output = tensor.concatenate(*h_list)

        if self.batch_first and x.ndim == 3:
            output = output.transpose(1, 0, 2)
        return output

    def __repr__(self) -> str:
        return "{}(input_size={}, hidden_size={}, batch_first={}, bidirectional={})".format(
            self.__class__.__name__, self.input_size, self.hidden_size,
            self.batch_first, self.bidirectional)