import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

try:
    import seaborn as sns
    sns.set()
except:
    pass

import pydynet.nn as nn
import pydynet.nn.functional as F
from pydynet.optim import Adam
from pydynet.data import data_loader
from pydynet.tensor import Tensor

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
train_X = stder.fit_transform(train_X)
test_X = stder.transform(test_X)

n_input = train_X.shape[1]
n_hidden = 1024
n_output = train_y.shape[1]


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


net = Net().to('cuda')
print(net)

optim = Adam(net.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 128

train_X = Tensor(train_X)
train_label = train_y.argmax(1)
train_y = Tensor(train_y)
test_X = Tensor(test_X)
test_label = test_y.argmax(1)
loader = data_loader(train_X, train_y, BATCH_SIZE, True)

for epoch in range(EPOCHES):
    net.train()
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to('cuda'), batch_y.to('cuda')
        output = net(batch_X)
        l = loss(output, batch_y)
        optim.zero_grad()
        l.backward()
        optim.step()