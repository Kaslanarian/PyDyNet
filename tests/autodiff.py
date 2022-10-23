import pydynet as pdn
import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set()
except:
    pass

x = pdn.Tensor(1., requires_grad=True)
lr = 2

x_list = []
y_list = []
for i in range(20):
    x_list.append(x.data.copy())
    y = pdn.log(pdn.square(x - 7) + 10)
    y_list.append(y.data)
    x.zero_grad()
    y.backward()
    x.data -= lr * x.grad
    if i % 5 == 4:
        print("Epoch {:2d}, y : {:.6f}".format(i + 1, y.data))

x = np.linspace(0, 10, 101)
plt.plot(x, np.log((x - 7)**2 + 10), label="$f(x)=\log((x-7)^2+10)$")
plt.scatter(x_list, y_list, color='orange')
plt.legend()
plt.savefig("../src/autodiff.png")
