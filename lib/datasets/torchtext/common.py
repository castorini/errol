import torch
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.vocab import Vectors

from datasets.dataset import Dataset
from utils.preprocessing import text_tokenize


class TorchtextDataset(Dataset):
    def __init__(self):
        self.doc_id_map = dict()
        self.train_examples = list()
        self.dev_examples = list()
        self.test_examples = list()

        self.label_field = Field(sequential=False, use_vocab=False, batch_first=True)
        self.logit_field = Field(sequential=False, use_vocab=False, batch_first=True)
        self.query_id_field = Field(sequential=False, use_vocab=False, batch_first=True)
        self.doc_id_field = Field(sequential=False, use_vocab=False, batch_first=True)
        self.query_field = Field(batch_first=True, tokenize=text_tokenize, include_lengths=True)
        self.input_field = Field(batch_first=True, tokenize=text_tokenize, include_lengths=True)
        self.fields = [('label', self.label_field), ('logit', self.logit_field), ('query_id', self.query_id_field),
                       ('doc_id', self.doc_id_field), ('query', self.query_field), ('input', self.input_field)]

    def get_splits(self, vectors_name, vectors_cache, device, batch_size):
        train_dataset = torchtext.data.Dataset(self.train_examples, self.fields)
        train_dataset.sort_key = lambda example: len(example.text)

        dev_dataset = torchtext.data.Dataset(self.dev_examples, self.fields)
        dev_dataset.sort_key = lambda example: len(example.text)

        test_dataset = torchtext.data.Dataset(self.test_examples, self.fields)
        test_dataset.sort_key = lambda example: len(example.text)

        vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=torch.Tensor.zero_)
        self.query_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=vectors)
        self.input_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=vectors)

        return BucketIterator.splits((train_dataset, dev_dataset, test_dataset), batch_size=batch_size,
                                     repeat=False, shuffle=True, sort_within_batch=True, device=device)
