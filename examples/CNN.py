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
from pydynet.dataloader import train_loader

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


class CNN1d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 1, 3)
        self.fc1 = nn.Linear(341, 128)
        self.fc2 = nn.Linear(128, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool1d(x, 12, 12)
        x = x.reshape(x.shape[0], -1)
        x = F.leaky_relu(self.fc1(x), 0.1)
        return self.fc2(x)


class CNN2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 4, 4)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.1)
        return self.fc2(x)


net1 = DNN()
net2 = CNN1d()
net3 = CNN2d()
optim1 = Adam(net1.parameters(), lr=0.01)
optim2 = Adam(net2.parameters(), lr=0.01)
optim3 = Adam(net3.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 32
loss_list1, loss_list2, loss_list3 = [], [], []
train_acc1, train_acc2, train_acc3 = [], [], []
test_acc1, test_acc2, test_acc3 = [], [], []

for epoch in range(EPOCHES):
    # 相同数据训练3个网络
    net1.train()
    net2.train()
    net3.train()
    for batch_X, batch_y in train_loader(
            train_X,
            train_y,
            batch_size=BATCH_SIZE,
            shuffle=True,
    ):
        node_y = Tensor(batch_y)

        output1 = net1(Tensor(batch_X))
        l1 = loss(output1, node_y)
        optim1.zero_grad()
        l1.backward()
        optim1.step()

        output2 = net2(Tensor(batch_X[:, np.newaxis, :]))
        l2 = loss(output2, node_y)
        optim2.zero_grad()
        l2.backward()
        optim2.step()

        output3 = net3(Tensor(batch_X.reshape(-1, 1, 64, 64)))
        l3 = loss(output3, node_y)
        optim3.zero_grad()
        l3.backward()
        optim3.step()

    net1.eval()
    net2.eval()
    net3.eval()

    output1 = net1(Tensor(train_X))
    output2 = net2(Tensor(train_X[:, np.newaxis, :]))
    output3 = net3(Tensor(train_X.reshape(-1, 1, 64, 64)))
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
            np.argmax(net2(Tensor(test_X[:, np.newaxis, :])).data, axis=1),
            np.argmax(test_y, axis=1),
        ))
    test_acc3.append(
        accuracy_score(
            np.argmax(net3(Tensor(test_X.reshape(-1, 1, 64, 64))).data,
                      axis=1),
            np.argmax(test_y, axis=1),
        ))

    if epoch % 10 in {4, 9}:
        print("Epoch {:2d}:".format(epoch + 1))
        print("DNN   : train loss {:6f} train acc {:.4f}, test acc {:.4f}".
              format(
                  loss_list1[-1],
                  train_acc1[-1] * 100,
                  test_acc1[-1] * 100,
              ))
        print("CNN1d : train loss {:6f} train acc {:.4f}, test acc {:.4f}".
              format(
                  loss_list2[-1],
                  train_acc2[-1] * 100,
                  test_acc2[-1] * 100,
              ))
        print("CNN2d : train loss {:6f} train acc {:.4f}, test acc {:.4f}".
              format(
                  loss_list3[-1],
                  train_acc3[-1] * 100,
                  test_acc3[-1] * 100,
              ))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_list1, label="Loss of DNN")
plt.plot(loss_list2, label="Loss of CNN1d")
plt.plot(loss_list3, label="Loss of CNN2d")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc1, label="Train Acc of DNN")
plt.plot(test_acc1, label="Test Acc of DNN")
plt.plot(train_acc2, label="Train Acc of CNN1d")
plt.plot(test_acc2, label="Test Acc of CNN1d")
plt.plot(train_acc3, label="Train Acc of CNN2d")
plt.plot(test_acc3, label="Test Acc of CNN2d")
plt.legend()

plt.savefig("../src/CNN.png")
