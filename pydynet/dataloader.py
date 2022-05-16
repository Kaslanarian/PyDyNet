import numpy as np
from math import ceil

from sklearn.svm import LinearSVC


def train_loader(X: np.ndarray,
                 y: np.ndarray,
                 batch_size: int,
                 shuffle: bool = False) -> list:
    '''对训练集数据进行划分，实现mini-batch机制。

    当测试数据较大时也可用该函数进行处理。
    
    Parameters
    ----------
    X : numpy.ndarray
        训练集特征数据
    y : numpy.ndarray
        训练集标签数据
    batch_size : int
        划分数据的大小
    shuffle : bool, default=False
        是否打乱，默认不打乱
    
    Return
    ------
    list
        以(特征数据, 标签数据)为单位的列表

    Example
    -------
    >>> import numpy as np
    >>> X = np.arange(4 * 3).reshape(4, 3)
    >>> y = np.array([0, 0, 1, 1])
    >>> loader = train_loader(X, y, batch_size=2)
    >>> for batch_X, batch_y in loader:
    ...     print(batch_X, batch_y)
    [[0 1 2]
    [3 4 5]] [0  0]
    [[ 6  7  8]
    [ 9 10 11]] [1  1]
    '''
    l = X.shape[0]
    if shuffle:
        order = np.random.choice(range(l), l, replace=False)
    else:
        order = range(l)
    X_split = np.array_split(X[order], ceil(l / batch_size))
    y_split = np.array_split(y[order], ceil(l / batch_size))
    return list(zip(X_split, y_split))
