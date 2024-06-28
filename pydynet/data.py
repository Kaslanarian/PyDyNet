from numpy.random import permutation


class Dataset:

    def __init__(self) -> None:
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Sampler:

    def __init__(self, dataset: Dataset) -> None:
        pass

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self) -> int:
        return len(self.dataset)


class RandomSampler(Sampler):

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        yield from permutation(len(self.dataset)).tolist()

    def __len__(self):
        return len(self.dataset)


class BatchSampler(Sampler):

    def __init__(self, sampler: Sampler, batch_size: int,
                 drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class _DataLoaderIter:

    def __init__(self, loader) -> None:
        self.loader = loader
        self.sample_iter = iter(self.loader.batch_sampler)

    def __next__(self):
        index = next(self.sample_iter)
        return self.loader.dataset[index]


class DataLoader:

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        return _DataLoaderIter(self)


def data_loader(X, y, batch_size: int, shuffle: bool = False) -> list:

    class TrainSet(Dataset):

        def __init__(self, X, y) -> None:
            self.data = X
            self.target = y

        def __getitem__(self, index):
            return self.data[index], self.target[index]

        def __len__(self):
            return len(self.data)

    return DataLoader(TrainSet(X, y), batch_size, shuffle)
