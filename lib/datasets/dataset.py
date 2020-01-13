import abc


class Dataset(abc.ABC):
    @abc.abstractmethod
    def get_splits(self, *args, **kwargs):
        raise NotImplementedError()
