import numpy as np
from math import ceil


def train_loader(X, y, batch_size, shuffle=False) -> list:
    '''
    对训练集数据进行划分，实现mini-batch机制
    
    parameters
    ----------
    X : 训练集特征数据
    y : 训练集标签数据
    batch_size : 划分数据的大小
    shuffle : 是否打乱
    
    return
    ------
    以(特征数据, 标签数据)为单位的列表
    '''
    l = X.shape[0]
    if shuffle:
        order = np.random.choice(range(l), l, replace=False)
    else:
        order = range(l)
    X_split = np.array_split(X[order], ceil(l / batch_size))
    y_split = np.array_split(y[order], ceil(l / batch_size))
    return list(zip(X_split, y_split))
