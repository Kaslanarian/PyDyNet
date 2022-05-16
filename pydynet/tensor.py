import numpy as np


class Graph:
    '''计算图，全局共用一个动态计算图'''
    node_list: list = list()

    @classmethod
    def add_node(cls, node):
        '''添加静态图节点'''
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        '''清空计算图'''
        cls.node_list.clear()

    @classmethod
    def free_graph(cls):
        '''
        释放计算图，和clear的区别在于我们不会删除叶子节点，
        这一点和PyTorch类似。
        '''
        new_list = []
        for node in Graph.node_list:
            node.next.clear()
            if node.is_leaf:
                # 叶子节点
                new_list.append(node)

            node.last.clear()
        Graph.node_list = new_list


class Tensor:
    '''
    将数据(NumPy数组)包装成可微分张量

    Parameters
    ----------
    data : ndarray
        张量数据，只要是np.array能够转换的数据;
    requires_grad : bool, default=False
        是否需要求梯度;
    dtype : default=None
        数据类型，和numpy数组的dtype等价

    Attributes
    ----------
    data : numpy.ndarray
        核心数据，为NumPy数组;
    requires_grad : bool
        是否需要求梯度;
    grad : numpy.ndarray
        梯度数据，为和data相同形状的数组(初始化为全0);
    next : list[Tensor]
        下游节点列表；
    last : list[Tensor]
        上游节点列表.

    Example
    -------
    >>> import numpy as np
    >>> from pydynet.tensor import Tensor
    >>> x = Tensor(1., requires_grad=True)
    >>> y = Tensor([1, 2, 3], dtype=float)
    >>> z = Tensor(np.random.rand(3, 4))
    '''
    def __init__(self, data, requires_grad: bool = False, dtype=None) -> None:
        if isinstance(data, Tensor):
            data = data.data
        self.data: np.ndarray = np.array(data, dtype)
        self.requires_grad: bool = requires_grad
        assert not (
            self.requires_grad and self.dtype != float
            and self.dtype != complex
        ), "Only Tensors of floating point and complex dtype can require gradients"
        self.grad: np.ndarray = np.zeros_like(
            self.data) if self.requires_grad else None
        self.next: list = list()
        self.last: list = list()
        if self.requires_grad:
            # 不需要求梯度的节点不出现在动态计算图中
            Graph.add_node(self)

    @property
    def is_leaf(self):
        '''判断是否为叶节点:需要求导且无上游节点的节点为叶节点.'''
        return not self.requires_grad or len(self.last) == 0

    @property
    def shape(self):
        '''张量的形状，用法同NumPy.
        
        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[2, 2]]).shape
        (1, 2)
        '''
        return self.data.shape

    @property
    def ndim(self):
        '''张量的维度，用法同NumPy.
        
        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[2, 2]]).ndim
        2
        '''
        return self.data.ndim

    @property
    def dtype(self):
        '''张量的数据类型，用法同NumPy.

        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[2, 2]]).dtype
        dtype('int64')
        '''
        return self.data.dtype

    @property
    def size(self):
        '''张量的元素个数，用法同NumPy.

        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[1, 1]]).size
        2
        '''
        return self.data.size

    @property
    def T(self):
        '''返回张量的完全转置，下面两种写法等价:

        >>> y = x.T
        >>> y = F.transpose(x)
        '''
        return self.transpose()

    def astype(self, new_type):
        '''类型转换，我们不允许可求导节点的类型转换'''
        assert self.requires_grad
        self.data.astype(new_type)

    def reshape(self, *new_shape):
        return reshape(self, new_shape)

    def transpose(self, *axes):
        return transpose(self, axes)

    def max(self, axis=None):
        return max(self, axis)

    def mean(self, axis=None):
        return mean(self, axis)

    def sum(self, axis=None):
        return sum(self, axis)

    @property
    def T(self):
        return transpose(self)

    def build_edge(self, node):
        self.next.append(node)
        node.last.append(self)

    def __repr__(self) -> str:
        return "{}({}, requires_grad={})".format(
            "tensor",
            self.data,
            self.requires_grad,
        )

    def __add__(self, x):
        return add(self, x)

    def __radd__(self, x):
        return add(x, self)

    def __sub__(self, x):
        return sub(self, x)

    def __rsub__(self, x):
        return sub(x, self)

    def __mul__(self, x):
        return mul(self, x)

    def __rmul__(self, x):
        return mul(x, self)

    def __matmul__(self, x):
        return matmul(self, x)

    def __rmatmul__(self, x):
        return matmul(x, self)

    def __truediv__(self, x):
        return div(self, x)

    def __rtruediv__(self, x):
        return div(x, self)

    def __pow__(self, x):
        return pow(self, x)

    def __rpow__(self, x):
        return pow(x, self)

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return abs(self)

    def __getitem__(self, key):
        return get_slice(self, key)

    def __setitem__(self, key, value):
        '''
        重载了切片/索引赋值的操作，我们不允许self允许求导，否则将出现错误

        Parameters
        ----------
        key : 索引，支持NumPy的数字、切片和条件索引
        value : 值，可以是NumPy数字，也可以是数字

        Example
        -------
        >>> x = Tensor([1, 2, 3])
        >>> x[x <= 2] = 0
        >>> x
        <[0 0 3], int64, Tensor>
        '''
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(key, Tensor):
            key = key.data
        if not isinstance(value, Tensor):
            self.data[key] = value
        else:
            self.data[key] = value.data

    def __len__(self):
        return len(self.data)

    def __iadd__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data += other
        return self

    def __isub__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data -= other
        return self

    def __imul__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data *= other
        return self

    def __itruediv__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data /= other
        return self

    def __imatmul__(self, other):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if isinstance(other, Tensor):
            other = other.data
        self.data @= other
        return self

    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data < other)

    def __le__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data <= other)

    # 这里没有重载__eq__和__neq__是因为在RNN中这样的重载会引发问题
    def equal(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data == other)

    def inequal(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data != other)

    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data > other)

    def __ge__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data >= other)

    def backward(self, retain_graph=False):
        '''
        以节点为输出进行反向传播

        Parameters
        ----------
        retain_graph : bool, default=False
            是否保留计算图

        Example
        -------
        >>> from pydynet.tensor import Tensor
        >>> import pydynet.functional as F
        >>> x = Tensor(2., requires_grad=True)
        >>> y = x**2 + x - 1
        >>> y.backward()
        >>> x.grad
        5.
        '''
        if self not in Graph.node_list:
            print("AD failed because the node is not in graph")
            return

        assert self.data.ndim == 0, "backward should be called only on a scalar"

        self.grad = np.ones_like(self.data)
        for i in range(len(Graph.node_list) - 1, -1, -1):
            if Graph.node_list[i] is self:
                y_id = i
                break

        for node in Graph.node_list[y_id::-1]:
            grad = node.grad
            for last in [l for l in node.last if l.requires_grad]:
                add_grad = node.grad_fn(last, grad)
                # 广播机制处理梯度
                if add_grad.shape != last.shape:
                    add_grad = np.sum(
                        add_grad,
                        axis=tuple(-i for i in range(1, last.ndim + 1)
                                   if last.shape[-i] == 1),
                        keepdims=True,
                    )
                    add_grad = np.sum(
                        add_grad,
                        axis=tuple(range(add_grad.ndim - last.ndim)),
                    )
                last.grad += add_grad

            if not node.is_leaf:
                node.grad = None

        if not retain_graph:
            Graph.free_graph()

    def zero_grad(self):
        '''梯度归零'''
        self.grad = np.zeros(self.shape)


class UnaryOperator(Tensor):
    '''
    一元运算算子的基类，将一个一元函数抽象成类
    '''
    def __init__(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        super().__init__(
            self.forward(x),
            x.requires_grad,
        )
        x.build_edge(self)

    def forward(self, x: Tensor) -> np.ndarray:
        '''
        前向传播函数，参数为Tensor，返回的是NumPy数组
        '''

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        '''
        x : Tensor
            上游节点
        grad : ndarray
            下游流入该节点的梯度
        '''

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.data)


class BinaryOperator(Tensor):
    '''
    二元运算算子的基类，将一个一元函数抽象成类
    '''
    def __init__(self, x: Tensor, y: Tensor) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        if not isinstance(y, Tensor):
            y = Tensor(y)
        super().__init__(
            self.forward(x, y),
            x.requires_grad or y.requires_grad,
        )
        x.build_edge(self)
        y.build_edge(self)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        pass

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        pass

    def __repr__(self) -> str:
        return "{}({})".format(self.__class__.__name__, self.data)


class add(BinaryOperator):
    '''
    加法算子

    Example
    -------
    >>> x = Tensor(1.)
    >>> y = Tensor(2.)
    >>> z = add(x, y) # 在Tensor类中进行了重载，所以也可以写成
    >>> z = x + y
    '''
    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        return np.ones(self.shape) * grad


class sub(BinaryOperator):
    '''
    减法算子，在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''
    def forward(self, x: Tensor, y: Tensor):
        return x.data - y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        grad_out = np.ones(self.shape) * grad
        if node is self.last[0]:
            return grad_out
        return -grad_out


class mul(BinaryOperator):
    '''
    元素级乘法算子，在Tensor类中进行重载

    Example
    -------
    ```python
    x = Tensor([1., 2.])
    y = Tensor([2., 3.])
    z = mul(x, y) # [2, 6]
    ```

    See also
    --------
    add : 加法算子
    '''
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.last[0]:
            return self.last[1].data * grad
        return self.last[0].data * grad


class div(BinaryOperator):
    '''
    除法算子，在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data / y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        temp = grad / self.last[1].data
        if node is self.last[0]:
            return temp
        return -self.data * temp


class pow(BinaryOperator):
    '''
    幂运算算子，在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data**y.data

    def grad_fn(self, node: Tensor, grad) -> np.ndarray:
        if node is self.last[0]:
            return (self.data * self.last[1].data / node.data) * grad
        else:
            return self.data * np.log(self.last[0].data) * grad


class matmul(BinaryOperator):
    '''
    矩阵乘法算子，在Tensor类中进行重载，张量的矩阵乘法遵从NumPy Matmul的规则.

    Reference
    ---------
    https://welts.xyz/2022/04/26/broadcast/

    See also
    --------
    add : 加法算子
    '''
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return x.data @ y.data

    def grad_fn(self, node: Tensor, grad) -> np.ndarray:
        if node is self.last[0]:
            if self.last[1].ndim == 1:
                return np.expand_dims(grad, -1) @ np.expand_dims(
                    self.last[1].data, -2)
            elif self.last[1].ndim > 2:
                shape = list(range(self.last[1].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return grad @ self.last[1].data.transpose(*shape)
            return grad @ self.last[1].data.T
        else:
            if self.last[0].ndim == 1:
                return np.expand_dims(self.last[0].data, -1) @ np.expand_dims(
                    grad, -2)
            elif self.last[0].ndim > 2:
                shape = list(range(self.last[0].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return self.last[0].data.transpose(*shape) @ grad
            return self.last[0].data.T @ grad


class abs(UnaryOperator):
    '''
    绝对值算子，在Tensor类中进行重载

    See also
    --------
    add : 加法算子
    '''
    def forward(self, x: Tensor) -> np.ndarray:
        return np.abs(x)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        mask = np.zeros(x.shape)
        mask[x > 0] = 1.
        mask[x < 0] = -1.
        return grad * mask


class sum(UnaryOperator):
    '''
    求和算子，在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求和方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    Example
    -------
    ```python
    x = Tensor(
        [[1, 2, 3],
        [4, 5, 6]]
    )
    s1 = x.sum(0) # [5, 7, 9]
    s2 = x.sum(1) # [6, 15]
    s3 = sum(x, keepdims=True) # [[21]]
    ```
    '''
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        return np.ones(x.shape) * grad


class mean(UnaryOperator):
    '''
    求均值算子，在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求均值方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    See also
    --------
    sum : 求和算子
    '''
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        return np.ones(x.shape) * grad * self.data.size / x.data.size


class max(UnaryOperator):
    '''
    求最大值算子，在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求最大值方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    See also
    --------
    sum : 求和算子
    '''
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        if self.keepdims:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = np.expand_dims(self.data, axis=self.axis)
            grad = np.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


# 非计算函数
class reshape(UnaryOperator):
    '''
    张量形状变换算子，在Tensor中进行重载

    Parameters
    ----------
    new_shape : tuple
        变换后的形状，用法同NumPy
    '''
    def __init__(self, x: Tensor, new_shape: tuple) -> None:
        self.new_shape = new_shape
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.reshape(self.new_shape)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return grad.reshape(x.shape)


class transpose(UnaryOperator):
    '''
    张量转置算子，在Tensor中进行重载(Tensor.T和Tensor.transpose)

    Parameters
    ----------
    axes : tuple
        转置的轴变换，用法同NumPy
    '''
    def __init__(self, x: Tensor, axes: tuple = None) -> None:
        self.axes = axes
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.transpose(self.axes)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        if self.axes is None:
            return grad.transpose()
        return grad.transpose(np.argsort(self.axes))


class get_slice(UnaryOperator):
    '''
    切片算子，为Tensor类提供索引和切片接口

    Example
    -------
    >>> x = Tensor(
            np.arange(12).reshape(3, 4).astype(float),
            requires_grad=True,
        )
    >>> y = x[:2, :2].sum()
    >>> y.backward()
    >>> x.grad 
    [[1. 1. 0. 0.]
     [1. 1. 0. 0.]
     [0. 0. 0. 0.]]
    '''
    def __init__(self, x: Tensor, key) -> None:
        if isinstance(key, Tensor):
            self.key = key.data
        else:
            self.key = key
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data[self.key]

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        full_grad = np.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


# 一些包装的特殊矩阵
def zeros(shape, requires_grad=False):
    return Tensor(np.zeros(shape), requires_grad=requires_grad)


def ones(shape, requires_grad=False):
    return Tensor(np.ones(shape), requires_grad=requires_grad)


def randn(*shape, requires_grad=False):
    return Tensor(np.random.randn(*shape), requires_grad=requires_grad)


def rand(*shape, requires_grad=False):
    return Tensor(np.random.rand(*shape), requires_grad=requires_grad)


def uniform(low, high, shape=None, requires_grad=False):
    return Tensor(
        np.random.uniform(low, high, size=shape),
        requires_grad=requires_grad,
    )
