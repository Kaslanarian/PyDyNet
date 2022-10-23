from tqdm import tqdm
from pydynet.tensor import Tensor
import pydynet.nn as nn
from pydynet.optim import Adam
from pydynet.data import data_loader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import seaborn as sns
    sns.set()
except:
    pass

np.random.seed(42)

# 数据预处理：独热化+标准化
data_X, data_y = load_digits(return_X_y=True)
data_y = OneHotEncoder(sparse=False).fit_transform(data_y.reshape(-1, 1))
train_X, test_X, train_y, test_y = train_test_split(
    data_X,
    data_y,
    train_size=0.7,
)
stder = StandardScaler()
train_X = stder.fit_transform(train_X).reshape(-1, 8, 8)
test_X = stder.transform(test_X).reshape(-1, 8, 8)


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=8,
            hidden_size=32,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x, hn = self.rnn(x)
        x = self.fc(hn.reshape(hn.shape[0], -1))
        return x


net = Net()
print(net)
optim = Adam(net.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 32
loader = data_loader(train_X, train_y, BATCH_SIZE, True)

loss_list, train_acc, test_acc = [], [], []

for epoch in tqdm(range(EPOCHES), desc="Training"):
    net.train()
    for batch_X, batch_y in loader:
        output = net(Tensor(batch_X))
        l = loss(output, batch_y)
        optim.zero_grad()
        l.backward()
        optim.step()

    net.eval()
    output = net(Tensor(train_X))
    loss_list.append(loss(output, train_y).data)
    train_acc.append(
        accuracy_score(
            np.argmax(output.data, axis=1),
            np.argmax(train_y, axis=1),
        ))
    test_acc.append(
        accuracy_score(
            np.argmax(net(Tensor(test_X)).data, axis=1),
            np.argmax(test_y, axis=1),
        ))
print(
    "epoch {:3d}, train loss {:.6f}, train acc {:.4f}, test acc {:.4f}".format(
        epoch + 1,
        loss_list[-1],
        train_acc[-1] * 100,
        test_acc[-1] * 100,
    ))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Cross Entropy Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.legend()

plt.savefig("../src/RNN.png")
