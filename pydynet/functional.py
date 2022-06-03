from ast import Str
from .tensor import BinaryOperator, Tensor, UnaryOperator
from .tensor import add, sub, mul, div, matmul, abs, sum, mean, max, reshape, transpose
import numpy as np


class exp(UnaryOperator):
    '''指数运算
    
    Example
    -------
    >>> x = Tensor(1.)
    >>> y = exp(x)
    '''
    def forward(self, x: Tensor):
        return np.exp(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * grad


class log(UnaryOperator):
    '''对数运算
    
    Example
    -------
    >>> x = Tensor(1.)
    >>> y = log(x)
    '''
    def forward(self, x: Tensor):
        return np.log(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad / x.data


class maximum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.maximum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x.data) * grad


class minimum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.minimum(x, y)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad


class sigmoid(UnaryOperator):
    '''Sigmoid运算，我们前向传播避免了溢出问题'''
    def forward(self, x: Tensor) -> np.ndarray:
        sigmoid = np.zeros(x.shape)
        sigmoid[x.data > 0] = 1 / (1 + np.exp(-x.data[x.data > 0]))
        sigmoid[x.data <= 0] = 1 - 1 / (1 + np.exp(x.data[x.data <= 0]))
        return sigmoid

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * (1 - self.data) * grad


class tanh(UnaryOperator):
    '''Tanh运算，我们前向传播避免了溢出问题'''
    def forward(self, x: Tensor) -> np.ndarray:
        tanh = np.zeros(x.shape)
        tanh[x.data > 0] = 2 / (1 + np.exp(-2 * x.data[x.data > 0])) - 1
        tanh[x.data <= 0] = 1 - 2 / (1 + np.exp(2 * x.data[x.data <= 0]))
        return tanh

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return (1 - self.data**2) * grad


def relu(x: Tensor):
    return maximum(0., x)


def leaky_relu(x: Tensor, alpha: float):
    return maximum(x, alpha * x)


def sqrt(x: Tensor):
    '''平方根函数'''
    return x**0.5


def square(x: Tensor):
    '''平方函数'''
    return x * x


def softmax(x: Tensor, axis=None, keepdims=False):
    '''Softmax函数'''
    x_sub_max = x - Tensor(np.ones(x.shape) * np.max(x.data))
    exp_ = exp(x_sub_max)
    return exp_ / sum(exp_, axis=axis, keepdims=keepdims)


def log_softmax(x: Tensor, axis=None, keepdims=False):
    '''log-softmax函数'''
    x_sub_max = x - Tensor(np.ones(x.shape) * np.max(x.data))
    return x_sub_max - log(sum(exp(x_sub_max), axis=axis, keepdims=keepdims))


# 卷积相关
class __im2col1d(UnaryOperator):
    def __init__(self, x: Tensor, kernel_size: int, stride: int) -> None:
        self.N, self.in_channels, self.n_features = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_output = (self.n_features - self.kernel_size) // stride + 1
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        col = np.zeros(
            (self.N, self.in_channels, self.n_output, self.kernel_size))

        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            col[..., i] = x.data[..., i:i_max:self.stride]

        return col

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        grad_x = np.zeros((self.N, self.in_channels, self.n_features))
        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            grad_x[..., i:i_max:self.stride] += grad[..., i]

        return grad_x


class __pad1d(UnaryOperator):
    def __init__(self, x: Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.pad(x.data, [(0, 0), (0, 0),
                               (self.pad_width, self.pad_width)], 'constant')

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width]


def conv1d(x: Tensor, kernel: Tensor, padding: int = 0, stride: int = 1):
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
    return (col @ kernel.transpose(1, 2, 0)).sum(1).transpose(0, 2, 1)


def max_pool1d(x: Tensor, kernel_size: int, stride: int, padding=0):
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


class __im2col2d(UnaryOperator):
    def __init__(self, x: Tensor, kernel_size, stride: int) -> None:
        self.N, self.in_channels, self.n_h, self.n_w = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_h, self.out_w = (
            self.n_h - self.kernel_size) // self.stride + 1, (
                self.n_w - self.kernel_size) // self.stride + 1
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        col = np.zeros((self.N, self.in_channels, self.kernel_size,
                        self.kernel_size, self.out_h, self.out_w))
        for i in range(self.kernel_size):
            i_max = i + self.out_h * self.stride
            for j in range(self.kernel_size):
                j_max = j + self.out_w * self.stride
                col[:, :, i, j, :, :] = x.data[:, :, i:i_max:self.stride,
                                               j:j_max:self.stride]

        return col

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        grad_col = grad
        grad_x = np.zeros((self.N, self.in_channels, self.n_h, self.n_w))
        for i in range(self.kernel_size):
            i_max = i + self.out_h * self.stride
            for j in range(self.kernel_size):
                j_max = j + self.out_w * self.stride
                grad_x[:, :, i:i_max:self.stride,
                       j:j_max:self.stride] = grad_col[:, :, i, j, :, :]
        return grad_x


class __pad2d(UnaryOperator):
    def __init__(self, x: Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.pad(x.data, [(0, 0), (0, 0),
                               (self.pad_width, self.pad_width),
                               (self.pad_width, self.pad_width)], 'constant')

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width,
                    self.pad_width:-self.pad_width]


def conv2d(x: Tensor, kernel: Tensor, padding: int = 0, stride: int = 1):
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


def max_pool2d(x: Tensor, kernel_size: int, stride: int, padding=0):
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
    out = max(col, axis=1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


class concatenate(Tensor):
    '''对多个张量进行连接，用法类似于`numpy.concatenate`
    
    Parameters
    ----------
    *tensors : 
        待连接的张量：
    axis : default=0
        连接轴，默认是沿着第一个轴拼接.
    '''
    def __init__(self, *tensors, axis=0) -> None:
        requires_grad = False
        self.tensors = list(tensors)
        self.axis = axis
        self.indices = [0]
        for i in range(len(self.tensors)):
            if not isinstance(tensors[i], Tensor):
                self.tensors[i] = Tensor(tensors[i])
            requires_grad = requires_grad or self.tensors[i].requires_grad
            self.indices.append(self.indices[-1] +
                                self.tensors[i].shape[self.axis])

        super().__init__(self.forward(), requires_grad=requires_grad)
        for i in range(len(self.tensors)):
            self.tensors[i].build_edge(self)

    def forward(self):
        return np.concatenate([t.data for t in self.tensors], axis=self.axis)

    def grad_fn(self, x, grad: np.ndarray):
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * len(grad.shape)
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


def mse_loss(y_pred, y_true, reduction='mean'):
    '''均方误差'''
    square_sum = square(y_pred - y_true)
    if reduction == 'mean':
        return mean(square_sum)
    elif reduction == 'sum':
        return sum(square_sum)
    else:
        assert 0, "reduction must be mean or sum."


def nll_loss(y_pred, y_true, reduction='mean'):
    '''负对数似然'''
    nll = -y_pred * y_true
    if reduction == 'mean':
        return mean(nll)
    elif reduction == 'sum':
        return sum(nll)
    else:
        assert 0, "reduction must be mean or sum."


def cross_entropy_loss(y_pred, y_true, reduction='mean'):
    '''交叉熵损失'''
    update_y_pred = y_pred - np.max(y_pred.data)
    log_sum_exp = log(sum(exp(update_y_pred), 1, keepdims=True))
    nll = (log_sum_exp - update_y_pred) * y_true
    if reduction == 'mean':
        return mean(nll)
    elif reduction == 'sum':
        return sum(nll)
    else:
        assert 0, "reduction must be mean or sum."
