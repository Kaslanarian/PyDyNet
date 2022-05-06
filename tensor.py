import numpy as np

try:
    from graphviz import Digraph
except:
    Digraph = None


class Graph:
    epsilon = 1e-100
    node_list = list()

    @classmethod
    def add_node(cls, node):
        cls.node_list.append(node)

    @classmethod
    def rm_node(cls, node):
        cls.node_list.remove(node)

    @classmethod
    def clear(cls):
        cls.node_list.clear()

    @classmethod
    def free_graph(cls):
        new_list = []
        for node in Graph.node_list:
            node.next.clear()
            if node.is_leaf:
                # 叶子节点
                new_list.append(node)

            node.last.clear()
        Graph.node_list = new_list


class Tensor:
    def __init__(self, value, requires_grad=False) -> None:
        self.data: np.ndarray = np.array(value, dtype=np.float)
        self.requires_grad: bool = requires_grad
        self.grad: np.ndarray = np.zeros_like(
            self.data) if self.requires_grad else None
        self.retain_grad: bool = False
        self.next: list = list()
        self.last: list = list()
        if self.requires_grad:
            # 不需要求梯度的节点不出现在动态计算图中
            Graph.add_node(self)

    @property
    def is_leaf(self):
        return not self.requires_grad or len(self.last) == 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def reshape(self, *new_shape):
        return reshape(self, new_shape)

    def transpose(self, *axes):
        return transpose(self, axes)

    @property
    def T(self):
        return transpose(self)

    def build_edge(self, node):
        self.next.append(node)
        node.last.append(self)

    def __repr__(self) -> str:
        data_info = float(self.data) if self.ndim == 0 else self.shape
        type_info = str(type(self))[8:-2]
        return "<{}, {}>".format(
            self.data,
            type_info[type_info.rfind(".") + 1:],
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

    def __getitem__(self, key):
        return get_slice(self, key)

    def __setitem__(self, key, value):
        assert not self.requires_grad, "In-place operation is forbidden in node requires grad."
        if type(value) != Graph.Node:
            self.data[key] = value
        else:
            self.data[key] = value.data

    def backward(self, retain_graph=False):
        if self not in Graph.node_list:
            print("AD failed because the node is not in graph")
            return

        self.grad = np.ones_like(self.data)
        y_id = Graph.node_list.index(self)
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

            if not node.is_leaf and not node.retain_grad:
                node.grad = None

        if not retain_graph:
            Graph.free_graph()

    def zero_grad(self):
        self.grad = np.zeros(self.shape)


class UnaryOperator(Tensor):
    def __init__(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        super().__init__(
            self.forward(x),
            x.requires_grad,
        )
        x.build_edge(self)

    def forward(self, x: Tensor) -> np.ndarray:
        pass

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        '''
        x : 上游节点
        grad : 下游流入该节点的梯度
        '''


class BinaryOperator(Tensor):
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


class add(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        return np.ones(self.shape) * grad


class sub(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor):
        return x.data - y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        grad_out = np.ones(self.shape) * grad
        if node == self.last[0]:
            return grad_out
        return -grad_out


class mul(BinaryOperator):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node == self.last[0]:
            return self.last[1].data * grad
        return self.last[0].data * grad


class div(BinaryOperator):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data / y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        temp = grad / self.last[1].data
        if node == self.last[0]:
            return temp
        return -self.data * temp


class pow(BinaryOperator):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data**y.data

    def grad_fn(self, node: Tensor, grad) -> np.ndarray:
        if node == self.last[0]:
            return (self.data * self.last[1].data / node.data) * grad
        else:
            return self.data * np.log(self.last[0].data) * grad


class matmul(BinaryOperator):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return x.data @ y.data

    def grad_fn(self, node: Tensor, grad) -> np.ndarray:
        # x, y = self.last[0].data[...], self.last[1].data[...]
        # if x.ndim == 1:
        #     x = np.expand_dims(x, axis=0)
        # if y.ndim == 1:
        #     y = np.expand_dims(y, axis=1)
        if node == self.last[0]:
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


# 非计算函数
class reshape(UnaryOperator):
    def __init__(self, x: Tensor, new_shape: tuple) -> None:
        self.new_shape = new_shape
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.reshape(self.new_shape)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return grad.reshape(x.shape)


class transpose(UnaryOperator):
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
    def __init__(self, x: Tensor, key) -> None:
        self.key = key
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data[self.key]

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        full_grad = np.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


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
