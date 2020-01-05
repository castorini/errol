import abc


class Trainer(abc.ABC):
    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()
