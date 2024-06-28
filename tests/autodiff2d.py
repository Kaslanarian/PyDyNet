import sys

sys.path.append('../pydynet')

import pydynet as pdn
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set_theme()
except:
    pass

np.random.seed(42)
A = pdn.Tensor([
    [3, 1],
    [1, 2.],
])
b = pdn.Tensor([-1., 1])


def f(x):
    return x @ A @ x / 2 + b @ x


Xs, ys = [], []
x = pdn.randn(2, requires_grad=True)
lr = 1e-1
for i in range(10):
    obj = f(x)
    obj.backward()

    Xs.append(x.data.copy())
    ys.append(obj.item())
    x.data -= lr * x.grad
    x.zero_grad()

Xs, ys = np.array(Xs), np.array(ys)

xd, yd, zd = Xs[:, 0], Xs[:, 1], ys

fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.plot3D(xd, yd, zd)
ax1.scatter3D(xd, yd, zd, label=r'$f(x)=\frac{1}{2}x^\top Ax+b^\top x$')
plt.legend()
plt.savefig("src/ad2d.png")
