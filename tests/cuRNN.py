from tqdm import tqdm
from pydynet.tensor import Tensor
import pydynet.nn.functional as F
import pydynet.nn as nn
from pydynet.optim import Adam
from pydynet.data import data_loader

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
        x = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x


net = Net().cuda()
print(net)
optim = Adam(net.parameters(), lr=0.01)
loss = nn.CrossEntropyLoss()
EPOCHES = 50
BATCH_SIZE = 128
train_X = Tensor(train_X)
train_y = Tensor(train_y)
loader = data_loader(train_X, train_y, BATCH_SIZE, True)

loss_list, train_acc, test_acc = [], [], []

for epoch in tqdm(range(EPOCHES), desc="Training"):
    net.train()
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
        output = net(batch_X)
        l = loss(output, batch_y)
        optim.zero_grad()
        l.backward()
        optim.step()
