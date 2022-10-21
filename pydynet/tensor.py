from typing import Any, List, Tuple, Union
import numpy as np
from .cuda import Device
from .autograd import is_grad_enable, no_grad


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
    def __init__(
        self,
        data: Any,
        dtype=None,
        device: Union[Device, int, str, None] = None,
        requires_grad: bool = False,
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data

        if not isinstance(device, Device):
            device = Device(device)

        self.device: Device = device
        with self.device:
            self.data = self.xp.array(data, dtype)

        self.requires_grad: bool = requires_grad and is_grad_enable()
        assert not (
            self.requires_grad and self.dtype != float
        ), "Only Tensors of floating point dtype can require gradients!"
        self.grad = self.xp.zeros_like(
            self.data) if self.requires_grad else None

        self.next: List[Tensor] = list()
        self.last: List[Tensor] = list()

        if self.requires_grad:
            # 不需要求梯度的节点不出现在动态计算图中
            Graph.add_node(self)

    @property
    def is_leaf(self) -> bool:
        '''判断是否为叶节点:需要求导且无上游节点的节点为叶节点.'''
        return not self.requires_grad or len(self.last) == 0

    @property
    def shape(self) -> Tuple[int]:
        '''张量的形状，用法同NumPy.
        
        Example
        -------
        >>> from pydynet import Tensor
        >>> Tensor([[2, 2]]).shape
        (1, 2)
        '''
        return self.data.shape

    @property
    def ndim(self) -> int:
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
    def size(self) -> int:
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
        return self.transpose()

    def astype(self, new_type):
        '''类型转换，我们不允许可求导节点的类型转换'''
        assert not self.requires_grad
        self.data.astype(new_type)

    def reshape(self, *new_shape):
        return reshape(self, new_shape)

    def transpose(self, *axes):
        return transpose(self, axes if len(axes) != 0 else None)

    def swapaxes(self, axis1: int, axis2: int):
        return swapaxes(self, axis1, axis2)

    def vsplit(self, indices_or_sections: Union[int, Tuple]):
        return vsplit(self, indices_or_sections)

    def hsplit(self, indices_or_sections: Union[int, Tuple]):
        return hsplit(self, indices_or_sections)

    def dsplit(self, indices_or_sections: Union[int, Tuple]):
        return dsplit(self, indices_or_sections)

    def split(self, indices_or_sections: Union[int, Tuple], axis=0):
        return split(self, indices_or_sections, axis=0)

    def max(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return max(self, axis, keepdims)

    def min(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return min(self, axis, keepdims)

    def mean(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return mean(self, axis, keepdims)

    def sum(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return sum(self, axis, keepdims)

    def argmax(self, axis: Union[int, Tuple, None] = None):
        return argmax(self, axis)

    def argmin(self, axis: Union[int, Tuple, None] = None):
        return argmin(self, axis)

    def build_edge(self, node):
        '''构建两节点的有向边，正常不适用'''
        self.next.append(node)
        node.last.append(self)

    def __repr__(self) -> str:
        return "{}({}, requires_grad={}".format(
            "Tensor",
            self.data,
            self.requires_grad,
        ) + (", device={}".format(self.device)
             if self.device.device != "cpu" else "") + ")"

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

    def __len__(self) -> int:
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

    @no_grad()
    def __lt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data < other)

    @no_grad()
    def __le__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data <= other)

    # 这里没有重载__eq__和__neq__是因为在RNN中这样的重载会引发问题
    @no_grad()
    def eq(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data == other)

    @no_grad()
    def ne(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data != other)

    @no_grad()
    def __gt__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data > other)

    @no_grad()
    def __ge__(self, other):
        if isinstance(other, Tensor):
            other = other.data
        return Tensor(self.data >= other)

    def backward(self, retain_graph: bool = False):
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
            print("AD failed because the node is not in graph.")
            return

        assert self.data.ndim == 0, "backward should be called only on a scalar."

        self.grad = self.xp.ones_like(self.data)
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
                    add_grad = self.xp.sum(
                        add_grad,
                        axis=tuple(-i for i in range(1, last.ndim + 1)
                                   if last.shape[-i] == 1),
                        keepdims=True,
                    )
                    add_grad = self.xp.sum(
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
        self.grad = self.xp.zeros(self.shape)

    def numpy(self) -> np.ndarray:
        '''返回Tensor的内部数据，即NumPy数组(拷贝)'''
        return self.cpu().data.copy()

    def item(self):
        return self.data.item()

    def to(self, device):
        device = Device(device)
        if self.device == device:
            return self
        elif device.device == "cpu":  # cuda -> cpu
            return Tensor(self.data.get(), dtype=self.dtype, device=device)
        else:  # cpu -> cuda
            return Tensor(self.data, dtype=self.dtype, device=device)

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    @property
    def xp(self):
        return self.device.xp


class UnaryOperator(Tensor):
    '''
    一元运算算子的基类，将一个一元函数抽象成类

    Example
    -------
    >>> class exp(UnaryOperator):
            def forward(self, x: Tensor):
                return np.exp(x.data)
            def grad_fn(self, x: Tensor, grad) -> np.ndarray:
                return self.data * grad
    >>> x = Tensor(1.)
    >>> y = exp(x)
    '''
    def __init__(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.device = x.device
        super().__init__(
            data=self.forward(x),
            device=x.device,
            requires_grad=is_grad_enable() and x.requires_grad,
        )
        if self.requires_grad:
            x.build_edge(self)

    def forward(self, x: Tensor) -> np.ndarray:
        '''前向传播函数，参数为Tensor，返回的是NumPy数组'''
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        '''
        反向传播函数，参数为下游节点，从上游流入该节点梯度。
        注："上游"和"下游"针对的是反向传播，比如z = f(x, y)，x和y是z的下游节点.

        x : Tensor
            下游节点
        grad : ndarray
            上游流入该节点的梯度
        '''
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class BinaryOperator(Tensor):
    '''
    二元运算算子的基类，将一个二元函数抽象成类

    Example
    -------
    >>> add(BinaryOperator):
            def forward(self, x: Tensor, y: Tensor):
                return x.data + y.data
            def grad_fn(self, node: Tensor, grad: np.ndarray):
                return np.ones(self.shape) * grad
    >>> x = Tensor(1.)
    >>> y = Tensor(2.)
    >>> z = add(x, y)
    '''
    def __init__(self, x: Tensor, y: Tensor) -> None:
        if not isinstance(x, Tensor) and isinstance(y, Tensor):
            x = Tensor(x, device=y.device)
        elif isinstance(x, Tensor) and not isinstance(y, Tensor):
            y = Tensor(y, device=x.device)
        elif not (isinstance(x, Tensor) and isinstance(y, Tensor)):
            x, y = Tensor(x), Tensor(y)
        assert x.device == y.device
        self.device = x.device
        super().__init__(
            data=self.forward(x, y),
            device=x.device,
            requires_grad=is_grad_enable()
            and (x.requires_grad or y.requires_grad),
        )
        if self.requires_grad:
            x.build_edge(self)
            y.build_edge(self)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        '''前向传播函数，参数为Tensor，返回的是NumPy数组'''
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        '''
        反向传播函数，参数为下游节点，从上游流入该节点梯度。
        注："上游"和"下游"针对的是反向传播，比如z = f(x, y)，x和y是z的下游节点.

        x : Tensor
            下游节点
        grad : ndarray
            上游流入该节点的梯度
        '''
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


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
        return grad[...]


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
        if node is self.last[0]:
            return grad[...]
        return -grad


class mul(BinaryOperator):
    '''
    元素级乘法算子，在Tensor类中进行重载

    Example
    -------
    >>> x = Tensor([1., 2.])
    >>> y = Tensor([2., 3.])
    >>> z = mul(x, y) # [2, 6]

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

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.last[0]:
            return (self.data * self.last[1].data / node.data) * grad
        else:
            return self.data * self.xp.log(self.last[0].data) * grad


class matmul(BinaryOperator):
    '''
    矩阵乘法算子，在Tensor类中进行重载，张量的矩阵乘法遵从NumPy Matmul的规则.

    参考 : https://welts.xyz/2022/04/26/broadcast/

    See also
    --------
    add : 加法算子
    '''
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return x.data @ y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.last[0]:
            if self.last[1].ndim == 1:
                return self.xp.expand_dims(grad, -1) @ self.xp.expand_dims(
                    self.last[1].data, -2)
            elif self.last[1].ndim > 2:
                shape = list(range(self.last[1].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return grad @ self.last[1].data.transpose(*shape)
            return grad @ self.last[1].data.T
        else:
            if self.last[0].ndim == 1:
                return self.xp.expand_dims(self.last[0].data,
                                           -1) @ self.xp.expand_dims(grad, -2)
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
        return self.xp.abs(x)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        mask = self.xp.zeros(x.shape)
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
    >>> x = Tensor(
            [[1, 2, 3],
            [4, 5, 6]]
        )
    >>> s1 = x.sum(0) # [5, 7, 9]
    >>> s2 = x.sum(1) # [6, 15]
    >>> s3 = sum(x, keepdims=True) # [[21]]
    ```
    '''
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return self.xp.ones(x.shape) * grad


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
        return self.xp.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return self.xp.ones(x.shape) * grad * self.data.size / x.data.size


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
        return self.xp.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = self.xp.expand_dims(self.data, axis=self.axis)
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class min(UnaryOperator):
    '''
    求最小值算子，在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求最大值方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    See also
    --------
    max : 求最大值算子
    '''
    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.min(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = self.xp.expand_dims(self.data, axis=self.axis)
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class argmax(Tensor):
    def __init__(self, x: Tensor, axis=None) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.device = x.device
        super().__init__(self.forward(x), device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.argmax(x.data, axis=self.axis)


class argmin(Tensor):
    def __init__(self, x: Tensor, axis=None) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.device = x.device
        super().__init__(self.forward(x), device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.argmin(x.data, axis=self.axis)


class exp(UnaryOperator):
    '''指数运算
    
    Example
    -------
    >>> x = Tensor(1.)
    >>> y = exp(x)
    '''
    def forward(self, x: Tensor):
        return self.xp.exp(x.data)

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
        return self.xp.log(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad / x.data


class maximum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return self.xp.maximum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x.data) * grad


class minimum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return self.xp.minimum(x, y)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad


def sqrt(x: Tensor):
    '''平方根函数'''
    return x**0.5


def square(x: Tensor):
    '''平方函数'''
    return x * x


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

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
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

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(self.axes)))


class swapaxes(UnaryOperator):
    '''
    张量交换轴算子

    Parameters
    ----------
    axis1 : int
        第一个axis;
    axis2 : int
        第二个axis.
    '''
    def __init__(self, x: Tensor, axis1: int, axis2: int) -> None:
        self.axis1 = axis1
        self.axis2 = axis2
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.swapaxes(self.axis1, self.axis2)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad.swapaxes(self.axis1, self.axis2)


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

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        full_grad = self.xp.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


class concatenate(Tensor):
    '''对多个张量进行连接，用法类似于`numpy.concatenate`
    
    Parameters
    ----------
    tensors : 
        待连接的张量：
    axis : default=0
        连接轴，默认是沿着第一个轴拼接.
    '''
    def __init__(self, tensors: List[Tensor], axis=0) -> None:
        requires_grad = False
        self.tensors = tensors
        self.axis = axis
        self.indices = [0]

        for i in range(len(self.tensors)):
            assert isinstance(
                tensors[i],
                Tensor), "Concatenate elements in 'tensors' must be 'Tensor'"
            if i == 0:
                device = tensors[i].device
            else:
                assert tensors[i].device == device
            requires_grad = requires_grad or self.tensors[i].requires_grad
            self.indices.append(self.indices[-1] +
                                self.tensors[i].shape[self.axis])
        self.device = device
        super().__init__(self.forward(),
                         requires_grad=requires_grad and is_grad_enable(),
                         device=device)
        if self.requires_grad:
            for i in range(len(self.tensors)):
                self.tensors[i].build_edge(self)

    def forward(self):
        return self.xp.concatenate([t.data for t in self.tensors],
                                   axis=self.axis)

    def grad_fn(self, x, grad: np.ndarray):
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * grad.ndim
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


def vsplit(
    x: Tensor,
    indices_or_sections: Union[int, Tuple],
) -> List[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[0]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[0]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[st:end])

    return sub_tensors


def hsplit(
    x: Tensor,
    indices_or_sections: Union[int, Tuple],
) -> List[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[1]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[1]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[:, st:end])

    return sub_tensors


def dsplit(
    x: Tensor,
    indices_or_sections: Union[int, Tuple],
) -> List[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[2]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[2]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()

    sub_tensors = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(x[:, :, st:end])

    return sub_tensors


def split(
    x: Tensor,
    indices_or_sections: Union[int, Tuple],
    axis: int = 0,
) -> List[Tensor]:
    if not isinstance(x, Tensor):
        x = Tensor(x)

    if axis == 0 or axis == -x.ndim:
        return vsplit(x, indices_or_sections)
    elif axis == 1 or axis == -x.ndim + 1:
        return hsplit(x, indices_or_sections)
    elif axis == 2 or axis == -x.ndim + 2:
        return dsplit(x, indices_or_sections)

    try:
        len(indices_or_sections)
    except TypeError:
        sections = indices_or_sections
        N = x.shape[axis]
        assert N % sections == 0, 'array split does not result in an equal division'

    Ntotal = x.shape[axis]
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError(
                'number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] + extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = np.array(section_sizes, dtype=np.intp).cumsum()

    sub_tensors = []
    stensor = swapaxes(x, 0, axis)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_tensors.append(swapaxes(stensor[st:end], axis, 0))
    return sub_tensors


# 一些包装的特殊矩阵
def zeros(shape, dtype=None, device=None, requires_grad=False):
    '''全0张量
    
    Parameters
    ----------
    shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.zeros(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def ones(shape, dtype=None, device=None, requires_grad=False):
    '''全1张量
    
    Parameters
    ----------
    shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.ones(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def randn(*shape, dtype=None, device=None, requires_grad=False):
    '''0-1正态分布张量
    
    Parameters
    ----------
    *shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.random.randn(*shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def rand(*shape, dtype=None, device=None, requires_grad=False):
    '''[0, 1)均匀分布张量
    
    Parameters
    ----------
    *shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.random.rand(*shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def uniform(low: float,
            high: float,
            shape=None,
            dtype=None,
            device=None,
            requires_grad=False):
    '''均匀分布张量
    
    Parameters
    ----------
    low : float
        均匀分布下界;
    high : float
        均匀分布下界;
    *shape : 
        张量形状
    require_grad : bool, default=False
        是否需要求导
    '''
    return Tensor(np.random.uniform(low, high, size=shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def empty(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.empty(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)
