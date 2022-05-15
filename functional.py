from tensor import Tensor, UnaryOperator
import tensor
import numpy as np

add = tensor.add
sub = tensor.sub
mul = tensor.mul
div = tensor.div
matmul = tensor.matmul
abs = tensor.abs
sum = tensor.sum
mean = tensor.mean
max = tensor.max
reshape = tensor.reshape
transpose = tensor.transpose


class exp(UnaryOperator):
    def forward(self, x: Tensor):
        return np.exp(x.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return self.data * grad


class log(UnaryOperator):
    def forward(self, x: Tensor):
        return np.log(x.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return grad / x.data


class sigmoid(UnaryOperator):
    def forward(self, x: Tensor) -> np.ndarray:
        sigmoid = np.zeros(x.shape)
        sigmoid[x.data > 0] = 1 / (1 + np.exp(-x.data[x.data > 0]))
        sigmoid[x.data <= 0] = 1 - 1 / (1 + np.exp(x.data[x.data <= 0]))
        return sigmoid

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return self.data * (1 - self.data) * grad


class tanh(UnaryOperator):
    def forward(self, x: Tensor) -> np.ndarray:
        tanh = np.zeros(x.shape)
        tanh[x.data > 0] = 2 / (1 + np.exp(-2 * x.data[x.data > 0])) - 1
        tanh[x.data <= 0] = 1 - 2 / (1 + np.exp(2 * x.data[x.data <= 0]))
        return tanh

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (1 - self.data**2) * grad


class relu(UnaryOperator):
    def forward(self, x: Tensor) -> np.ndarray:
        return np.array([x.data, np.zeros(x.shape)]).max(0)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return grad * (x.data >= 0)


class leaky_relu(UnaryOperator):
    def __init__(self, x: Tensor, alpha: float) -> None:
        self.alpha = alpha
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.array([x.data, self.alpha * x.data]).max(0)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        dlrelu = np.zeros(self.shape)
        dlrelu[self.data >= 0] = 1.
        dlrelu[self.data < 0] = self.alpha
        return dlrelu * grad


def sqrt(x: Tensor):
    return x**0.5


def square(x: Tensor):
    return x * x


def softmax(x: Tensor, axis=None, keepdims=False):
    x_sub_max = x - Tensor(np.ones(x.shape) * np.max(x.data))
    exp_ = exp(x_sub_max)
    return exp_ / sum(exp_, axis=axis, keepdims=keepdims)


def log_softmax(x: Tensor, axis=None, keepdims=False):
    x_sub_max = x - Tensor(np.ones(x.shape) * np.max(x.data))
    return x_sub_max - log(sum(exp(x_sub_max), axis=axis, keepdims=keepdims))


# 卷积相关
class im2col1d(UnaryOperator):
    def __init__(self, x: Tensor, kernel_size: int, stride: int) -> None:
        self.N, self.in_channels, self.n_features = x.shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_output = (self.n_features - self.kernel_size) // stride + 1
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        col = np.zeros((
            self.N,
            self.in_channels,
            self.kernel_size,
            self.n_output,
        ))
        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            col[..., i, :] = x.data[..., i:i_max:self.stride]
        return col

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        grad_col = grad
        grad_x = np.zeros((self.N, self.in_channels, self.n_features))
        for i in range(self.kernel_size):
            i_max = i + self.n_output * self.stride
            grad_x[..., i:i_max:self.stride] = grad_col[..., i, :]
        return grad_x


class pad1d(UnaryOperator):
    def __init__(self, x: Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.pad(
            x.data,
            [(0, 0), (0, 0), (self.pad_width, self.pad_width)],
            'constant',
        )

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width]


def conv1d(x: Tensor, kernel: Tensor, padding: int = 0, stride: int = 1):
    N, _, _ = x.shape
    out_channels, _, kernel_size = kernel.shape
    pad_x = pad1d(x, padding)
    col = im2col1d(pad_x, kernel_size, stride)
    n_output = col.shape[-1]
    col = col.transpose(0, 3, 1, 2).reshape(N * n_output, -1)
    col_filter = kernel.reshape(out_channels, -1).T
    out = col @ col_filter
    return out.reshape(N, n_output, -1).transpose(0, 2, 1)


def max_pool1d(x: Tensor, kernel_size: int, stride: int, padding=0):
    N, out_channels, _ = x.shape
    pad_x = pad1d(x, padding)
    col = im2col1d(pad_x, kernel_size, stride)
    n_output = col.shape[-1]
    col = col.transpose(0, 3, 1, 2).reshape(-1, kernel_size)
    out = max(col, axis=1)
    out = out.reshape(N, n_output, out_channels).transpose(0, 2, 1)
    return out


class im2col2d(UnaryOperator):
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

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        grad_col = grad
        grad_x = np.zeros((self.N, self.in_channels, self.n_h, self.n_w))
        for i in range(self.kernel_size):
            i_max = i + self.out_h * self.stride
            for j in range(self.kernel_size):
                j_max = j + self.out_w * self.stride
                grad_x[:, :, i:i_max:self.stride,
                       j:j_max:self.stride] = grad_col[:, :, i, j, :, :]
        return grad_x


class pad2d(UnaryOperator):
    def __init__(self, x: Tensor, pad_width=0) -> None:
        self.pad_width = pad_width
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.pad(
            x.data,
            [(0, 0), (0, 0), (self.pad_width, self.pad_width),
             (self.pad_width, self.pad_width)],
            'constant',
        )

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        if self.pad_width == 0:
            return grad[...]
        return grad[..., self.pad_width:-self.pad_width,
                    self.pad_width:-self.pad_width]


def conv2d(x: Tensor, kernel: Tensor, padding: int = 0, stride: int = 1):
    N, _, _, _ = x.shape
    out_channels, _, kernel_size, _ = kernel.shape
    pad_x = pad2d(x, padding)
    col = im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    col_filter = kernel.reshape(out_channels, -1).T
    out = col @ col_filter
    return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)


def max_pool2d(x: Tensor, kernel_size: int, stride: int, padding=0):
    N, in_channels, _, _ = x.shape
    pad_x = pad2d(x, padding)
    col = im2col2d(pad_x, kernel_size, stride)
    out_h, out_w = col.shape[-2:]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(
        -1,
        kernel_size * kernel_size,
    )
    out = max(col, axis=1)
    out = out.reshape(N, out_h, out_w, in_channels).transpose(0, 3, 1, 2)
    return out


class concatenate(Tensor):
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

    def grad_fn(self, x, grad):
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * len(grad.shape)
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


def mse_loss(y_pred, y_true, reduction='mean'):
    square_sum = square(y_pred - y_true)
    if reduction == 'mean':
        return mean(square_sum)
    elif reduction == 'sum':
        return sum(square_sum)
    else:
        assert 0, "reduction must be mean or sum."


def nll_loss(y_pred, y_true, reduction='mean'):
    nll = -y_pred * y_true
    if reduction == 'mean':
        return mean(nll)
    elif reduction == 'sum':
        return sum(nll)
    else:
        assert 0, "reduction must be mean or sum."


def cross_entropy_loss(y_pred, y_true, reduction='mean'):
    update_y_pred = y_pred - np.max(y_pred.data)
    log_sum_exp = log(sum(exp(update_y_pred), 1, keepdims=True))
    nll = -(update_y_pred - log_sum_exp) * y_true
    if reduction == 'mean':
        return mean(nll)
    elif reduction == 'sum':
        return sum(nll)
    else:
        assert 0, "reduction must be mean or sum."
