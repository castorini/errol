import torch
import torchtext
from torchtext.data import Field, BucketIterator
from torchtext.vocab import Vectors

from common.constants import WORD2VEC_EMBEDDING_FILE, WORD2VEC_EMBEDDING_DIR
from datasets.dataset import Dataset
from utils.preprocessing import text_tokenize


class TorchtextDataset(Dataset):
    def __init__(self):
        self.doc_id_map = dict()
        self.train_examples = list()
        self.dev_examples = list()
        self.test_examples = list()

        self.label_field = Field(sequential=False, use_vocab=False, batch_first=True)
        self.logits_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
        self.query_id_field = Field(sequential=False, use_vocab=False, batch_first=True)
        self.doc_id_field = Field(sequential=False, use_vocab=False, batch_first=True)
        self.query_field = Field(batch_first=True, tokenize=text_tokenize, include_lengths=True)
        self.input_field = Field(batch_first=True, tokenize=text_tokenize, include_lengths=True)
        self.fields = [('label', self.label_field), ('logits', self.logits_field), ('query_id', self.query_id_field),
                       ('doc_id', self.doc_id_field), ('query', self.query_field), ('input', self.input_field)]

    def get_splits(self, device, batch_size):
        train_dataset = torchtext.data.Dataset(self.train_examples, self.fields)
        train_dataset.sort_key = lambda example: len(example.input)

        dev_dataset = torchtext.data.Dataset(self.dev_examples, self.fields)
        dev_dataset.sort_key = lambda example: len(example.input)

        test_dataset = torchtext.data.Dataset(self.test_examples, self.fields)
        test_dataset.sort_key = lambda example: len(example.input)

        vectors = Vectors(name=WORD2VEC_EMBEDDING_FILE, cache=WORD2VEC_EMBEDDING_DIR, unk_init=torch.Tensor.zero_)
        self.input_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=vectors)
        self.query_field.vocab = self.input_field.vocab

        return BucketIterator.splits((train_dataset, dev_dataset, test_dataset), batch_size=batch_size,
                                     repeat=False, shuffle=True, sort_within_batch=True, device=device)
