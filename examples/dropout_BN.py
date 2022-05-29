import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pydynet.tensor import Tensor
import pydynet.functional as F
import pydynet.nn as nn
from pydynet.optimizer import Adam
from pydynet.util import train_loader

try:
    import seaborn as sns
    sns.set()
except:
    pass

np.random.seed(0)

data_X, data_y = fetch_olivetti_faces(return_X_y=True)
data_y = OneHotEncoder(sparse=False).fit_transform(data_y.reshape(-1, 1))
train_X, test_X, train_y, test_y = train_test_split(
    data_X,
    data_y,
    train_size=0.7,
)
stder = StandardScaler()
train_X = stder.fit_transform(train_X)
test_X = stder.transform(test_X)


class DNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4096, 128)
        self.fc2 = nn.Linear(128, 40)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)


class DNN_dropout(DNN):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = F.sigmoid(self.dropout(self.fc1(x)))
        return self.fc2(x)


class DNN_BN(DNN):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm(128)

    def forward(self, x):
        x = self.bn(self.fc1(x))
        x = F.sigmoid(x)
        return self.fc2(x)


net1 = DNN()
net2 = DNN_dropout()
net3 = DNN_BN()
print(net1)
print(net2)
print(net3)
optim1 = Adam(net1.parameters(), lr=0.01)
optim2 = Adam(net2.parameters(), lr=0.01)
optim3 = Adam(net3.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 32

loader = train_loader(train_X, train_y, BATCH_SIZE, True)

loss_list1, loss_list2, loss_list3 = [], [], []
train_acc1, train_acc2, train_acc3 = [], [], []
test_acc1, test_acc2, test_acc3 = [], [], []

for epoch in range(EPOCHES):
    # 相同数据训练3个网络
    net1.train()
    net2.train()
    net3.train()
    for batch_X, batch_y in loader:
        node_y = Tensor(batch_y)

        output1 = net1(Tensor(batch_X))
        l1 = loss(output1, node_y)
        optim1.zero_grad()
        l1.backward()
        optim1.step()

        output2 = net2(Tensor(batch_X))
        l2 = loss(output2, node_y)
        optim2.zero_grad()
        l2.backward()
        optim2.step()

        output3 = net3(Tensor(batch_X))
        l3 = loss(output3, node_y)
        optim3.zero_grad()
        l3.backward()
        optim3.step()

    net1.eval()
    net2.eval()
    net3.eval()

    output1 = net1(Tensor(train_X))
    output2 = net2(Tensor(train_X))
    output3 = net3(Tensor(train_X))
    loss_list1.append(loss(output1, train_y).data)
    loss_list2.append(loss(output2, train_y).data)
    loss_list3.append(loss(output3, train_y).data)
    train_acc1.append(
        accuracy_score(
            np.argmax(output1.data, axis=1),
            np.argmax(train_y, axis=1),
        ))
    train_acc2.append(
        accuracy_score(
            np.argmax(output2.data, axis=1),
            np.argmax(train_y, axis=1),
        ))
    train_acc3.append(
        accuracy_score(
            np.argmax(output3.data, axis=1),
            np.argmax(train_y, axis=1),
        ))
    test_acc1.append(
        accuracy_score(
            np.argmax(net1(Tensor(test_X)).data, axis=1),
            np.argmax(test_y, axis=1),
        ))
    test_acc2.append(
        accuracy_score(
            np.argmax(net2(Tensor(test_X)).data, axis=1),
            np.argmax(test_y, axis=1),
        ))
    test_acc3.append(
        accuracy_score(
            np.argmax(net3(Tensor(test_X)).data, axis=1),
            np.argmax(test_y, axis=1),
        ))

    if epoch % 10 in {4, 9}:
        print("Epoch {:2d}:".format(epoch + 1))
        print("DNN    : train loss {:6f} train acc {:.4f}, test acc {:.4f}".
              format(
                  loss_list1[-1],
                  train_acc1[-1] * 100,
                  test_acc1[-1] * 100,
              ))
        print("Dropout: train loss {:6f} train acc {:.4f}, test acc {:.4f}".
              format(
                  loss_list2[-1],
                  train_acc2[-1] * 100,
                  test_acc2[-1] * 100,
              ))
        print("BN     : train loss {:6f} train acc {:.4f}, test acc {:.4f}".
              format(
                  loss_list3[-1],
                  train_acc3[-1] * 100,
                  test_acc3[-1] * 100,
              ))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list1, label="Loss of DNN")
plt.plot(loss_list2, label="Loss of DNN with dropout")
plt.plot(loss_list3, label="Loss of DNN with BN")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc1, label="Train Acc of DNN")
plt.plot(test_acc1, label="Test Acc of DNN")
plt.plot(train_acc2, label="Train Acc of DNN with dropout")
plt.plot(test_acc2, label="Test Acc of DNN with dropout")
plt.plot(train_acc3, label="Train Acc of DNN with BN")
plt.plot(test_acc3, label="Test Acc of DNN with BN")
plt.legend()

plt.savefig("../src/dropout_BN.png")
