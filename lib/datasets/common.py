import abc


class Dataset(abc.ABC):
    @abc.abstractmethod
    def get_splits(self, *args, **kwargs):
        raise NotImplementedError()


class Example(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, label, doc_id=None, query_id=None):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label = label
        self.doc_id = doc_id
        self.query_id = query_id
