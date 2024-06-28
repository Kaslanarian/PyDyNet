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

x = pdn.Tensor(1., requires_grad=True, device='cuda')
lr = 2

x_list = []
y_list = []
for i in range(20):
    x_list.append(x.item())
    y = pdn.log(pdn.square(x - 7) + 10)
    y_list.append(y.item())

    x.zero_grad()
    y.backward()

    x.data -= lr * x.grad

    print("Iter {:2d}, y : {:.6f}".format(i + 1, y.item()))

x = np.linspace(0, 10, 101)
plt.plot(x, np.log((x - 7)**2 + 10), label="$f(x)=\log((x-7)^2+10)$")
plt.scatter(x_list, y_list, color='orange')
plt.legend()
plt.savefig("src/ad1d.png")
