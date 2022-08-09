import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from pydynet.tensor import Tensor
import pydynet.nn.functional as F
import pydynet.nn as nn
from pydynet.optim import Adam
from pydynet.data import data_loader

try:
    import seaborn as sns
    sns.set()
except:
    pass

np.random.seed(42)

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


net1 = DNN().cuda()
net2 = CNN1d().cuda()
net3 = CNN2d().cuda()
print(net1)
print(net2)
print(net3)
optim1 = Adam(net1.parameters(), lr=0.01)
optim2 = Adam(net2.parameters(), lr=0.01)
optim3 = Adam(net3.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 64

loader = data_loader(Tensor(train_X), Tensor(train_y), BATCH_SIZE, True)

for epoch in range(EPOCHES):
    # 相同数据训练3个网络
    net2.train()
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to('cuda'), batch_y.to('cuda')
        output1 = net1(batch_X)
        l1 = loss(output1, batch_y)
        optim1.zero_grad()
        l1.backward()
        optim1.step()

        output2 = net2(batch_X[:, np.newaxis, :])
        l2 = loss(output2, batch_y)
        optim2.zero_grad()
        l2.backward()
        optim2.step()

        output3 = net3(batch_X.reshape(-1, 1, 64, 64))
        l3 = loss(output3, batch_y)
        optim3.zero_grad()
        l3.backward()
        optim3.step()
