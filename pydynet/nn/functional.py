import numpy as np
from typing import Tuple, Union

from .. import tensor


def size_handle(input: Union[int, Tuple[int, int]]):
    if type(input) is int:
        return input, input
    assert type(input) in {list, tuple} and len(input) == 2
    return input


def linear(x: tensor.Tensor, weight: tensor.Tensor, bias: tensor.Tensor):
    affine = x @ weight
    if bias is not None:
        affine = affine + bias
    return affine


class sigmoid(tensor.UnaryOperator):
    '''Sigmoid运算，我们前向传播避免了溢出问题'''
    def forward(self, x: tensor.Tensor) -> np.ndarray:
        sigmoid = self.xp.zeros(x.shape)
        sigmoid[x.data > 0] = 1 / (1 + self.xp.exp(-x.data[x.data > 0]))
        sigmoid[x.data <= 0] = 1 - 1 / (1 + self.xp.exp(x.data[x.data <= 0]))
        return sigmoid

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * (1 - self.data) * grad


class tanh(tensor.UnaryOperator):
    '''Tanh运算，我们前向传播避免了溢出问题'''
    def forward(self, x: tensor.Tensor) -> np.ndarray:
        tanh = self.xp.zeros(x.shape)
        tanh[x.data > 0] = 2 / (1 + self.xp.exp(-2 * x.data[x.data > 0])) - 1
        tanh[x.data <= 0] = 1 - 2 / (1 + self.xp.exp(2 * x.data[x.data <= 0]))
        return tanh

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        return (1 - self.data**2) * grad


def relu(x: tensor.Tensor):
    return tensor.maximum(0., x)


def leaky_relu(x: tensor.Tensor, alpha: float):
    return tensor.maximum(x, alpha * x)


def softmax(x: tensor.Tensor, axis=None, keepdims=False):
    '''Softmax函数'''
    x_sub_max = x - x.data.max()
    exp_ = tensor.exp(x_sub_max)
    return exp_ / tensor.sum(exp_, axis=axis, keepdims=keepdims)


def log_softmax(x: tensor.Tensor, axis=None, keepdims=False):
    '''log-softmax函数'''
    x_sub_max = x - x.data.max()
    return x_sub_max - tensor.log(
        tensor.sum(tensor.exp(x_sub_max), axis=axis, keepdims=keepdims))


class __im2col1d(tensor.UnaryOperator):
    def __init__(
        self,
        x: tensor.Tensor,
        kernel_size: int,
        stride: int,
    ) -> None:
        self.N, self.in_channels, self.n_features = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_output = (self.n_features - self.kernel_size) // stride + 1
        super().__init__(x)

    def forward(self, x: tensor.Tensor) -> np.ndarray:
        col = self.xp.zeros(
            (self.N, self.in_channels, self.n_output, self.kernel_size))

        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            col[..., i] = x.data[..., i:i_max:self.stride]

        return col

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        grad_x = self.xp.zeros((self.N, self.in_channels, self.n_features))
        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            grad_x[..., i:i_max:self.stride] += grad[..., i]

        return grad_x


class __pad1d(tensor.UnaryOperator):
    def __init__(self, x: tensor.Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward(self, x: tensor.Tensor) -> np.ndarray:
        return self.xp.pad(x.data, [(0, 0), (0, 0),
                                    (self.pad_width, self.pad_width)],
                           'constant')

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width]


def conv1d(
    x: tensor.Tensor,
    kernel: tensor.Tensor,
    padding: int = 0,
    stride: int = 1,
):
    '''一维卷积函数

    基于im2col实现的一维卷积.
    
    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_features);
    kernel : Tensor
        卷积核，形状为(out_channels, in_channels, kernel_size);
    padding : int, default=0
        对输入特征两边补0数量;
    stride : int, default=1
        卷积步长.
    '''
    kernel_size = kernel.shape[-1]
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return (col @ kernel.transpose(1, 2, 0)).sum(1).swapaxes(1, 2)


def max_pool1d(
    x: tensor.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 0,
):
    '''一维池化函数

    基于im2col实现的一维池化.`
    
    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_features);
    kernel_size : int
        池化核大小;
    stride : int
        卷积步长;
    padding : int, default=0
        对输入特征两边补0数量.
    '''
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return col.max(-1)


def avg_pool1d(
    x: tensor.Tensor,
    kernel_size: int,
    stride: int,
    padding: int = 0,
):
    '''一维平均池化函数

    基于im2col实现的一维池化.`
    
    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_features);
    kernel_size : int
        池化核大小;
    stride : int
        卷积步长;
    padding : int, default=0
        对输入特征两边补0数量.
    '''
    pad_x = __pad1d(x, padding)
    col = __im2col1d(pad_x, kernel_size, stride)
    return col.mean(-1)


class __im2col2d(tensor.UnaryOperator):
    def __init__(
        self,
        x: tensor.Tensor,
        kernel_size: int,
        stride: int,
    ) -> None:
        self.N, self.in_channels, self.n_h, self.n_w = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_h, self.out_w = (
            self.n_h - self.kernel_size) // self.stride + 1, (
                self.n_w - self.kernel_size) // self.stride + 1
        super().__init__(x)

    def forward(self, x: tensor.Tensor) -> np.ndarray:
        col = self.xp.zeros((self.N, self.in_channels, self.kernel_size,
                             self.kernel_size, self.out_h, self.out_w))
        for i in range(self.kernel_size):
            i_max = i + self.out_h * self.stride
            for j in range(self.kernel_size):
                j_max = j + self.out_w * self.stride
                col[:, :, i, j, :, :] = x.data[:, :, i:i_max:self.stride,
                                               j:j_max:self.stride]

        return col

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        grad_col = grad
        grad_x = self.xp.zeros((self.N, self.in_channels, self.n_h, self.n_w))
        for i in range(self.kernel_size):
            i_max = i + self.out_h * self.stride
            for j in range(self.kernel_size):
                j_max = j + self.out_w * self.stride
                grad_x[:, :, i:i_max:self.stride,
                       j:j_max:self.stride] = grad_col[:, :, i, j, :, :]
        return grad_x


class __pad2d(tensor.UnaryOperator):
    def __init__(self, x: tensor.Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward(self, x: tensor.Tensor) -> np.ndarray:
        return self.xp.pad(x.data, [(0, 0), (0, 0),
                                    (self.pad_width, self.pad_width),
                                    (self.pad_width, self.pad_width)],
                           'constant')

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width,
                    self.pad_width:-self.pad_width]


def conv2d(x: tensor.Tensor,
           kernel: tensor.Tensor,
           padding: int = 0,
           stride: int = 1):
    '''二维卷积函数

    基于im2col实现的二维卷积. 为了实现上的方便，我们不考虑长宽不同的卷积核，步长和补零。
    
    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_height, n_width);
    kernel : Tensor
        卷积核，形状为(out_channels, in_channels, kernel_height, kernel_width);
    padding : int, default=0
        对输入图片周围补0数量;
    stride : int, default=1
        卷积步长.
    '''
    N, _, _, _ = x.shape
    out_channels, _, kernel_size, _ = kernel.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    col_filter = kernel.reshape(out_channels, -1).T
    out = col @ col_filter
    return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)


def max_pool2d(x: tensor.Tensor, kernel_size: int, stride: int, padding=0):
    '''二维卷积函数池化

    基于im2col实现的二维卷积. 为了实现上的方便，我们不考虑长宽不同的kernel_size，步长和补零。
    
    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_height, n_width);
    kernel_size : int
        池化核尺寸;
    stride : int, default=1
        卷积步长;
    padding : int, default=0
        对输入图片周围补0数量;
    '''
    N, in_channels, _, _ = x.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(
        -1,
        kernel_size * kernel_size,
    )
    out = col.max(1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


def avg_pool2d(x: tensor.Tensor, kernel_size: int, stride: int, padding=0):
    '''二维平均池化

    基于im2col实现的二维池化. 为了实现上的方便，我们不考虑长宽不同的kernel_size，步长和补零。
    
    Parameters
    ----------
    x : Tensor
        输入数据，形状为(N, in_channels, n_height, n_width);
    kernel_size : int
        池化核尺寸;
    stride : int, default=1
        卷积步长;
    padding : int, default=0
        对输入图片周围补0数量;
    '''
    N, in_channels, _, _ = x.shape
    pad_x = __pad2d(x, padding)
    col = __im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(
        -1,
        kernel_size * kernel_size,
    )
    out = col.mean(1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


def mse_loss(y_pred, y_true, reduction='mean'):
    '''均方误差'''
    square_sum = tensor.square(y_pred - y_true)
    if reduction == 'mean':
        return tensor.mean(square_sum)
    elif reduction == 'sum':
        return tensor.sum(square_sum)
    else:
        assert 0, "reduction must be mean or sum."


def nll_loss(y_pred, y_true, reduction='mean'):
    '''负对数似然'''
    nll = -y_pred * y_true
    if reduction == 'mean':
        return tensor.mean(nll)
    elif reduction == 'sum':
        return tensor.sum(nll)
    else:
        assert 0, "reduction must be mean or sum."


def cross_entropy_loss(y_pred, y_true, reduction='mean'):
    '''交叉熵损失'''
    update_y_pred = y_pred - y_pred.data.max()
    log_sum_exp = tensor.log(
        tensor.sum(tensor.exp(update_y_pred), 1, keepdims=True))
    nll = (log_sum_exp - update_y_pred) * y_true
    if reduction == 'mean':
        return tensor.mean(nll)
    elif reduction == 'sum':
        return tensor.sum(nll)
    else:
        assert 0, "reduction must be mean or sum."
