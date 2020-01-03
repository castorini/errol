import ast
import csv
import os
import sys

import torch
from torchtext.data import Field, Example, Dataset
from torchtext.data.iterator import BucketIterator
from torchtext.vocab import Vectors

from lib.util.preprocessing import text_tokenize, binary_one_hot

# Set upper limit on parsed CSV fields
csv.field_size_limit(sys.maxsize)


class Robust04:
    NAME = 'Robust04'
    NUM_CLASSES = 2

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
        self.text_field = Field(batch_first=True, tokenize=text_tokenize, include_lengths=True)
        self.fields = [('label', self.label_field), ('logit', self.logit_field), ('query_id', self.query_id_field),
                       ('doc_id', self.doc_id_field), ('query', self.query_field), ('text', self.text_field)]

    def load_examples(self, dataset_path, dev_splits, test_splits):
        with open(dataset_path, 'r') as dataset_tsv:
            for line in dataset_tsv:
                data_row = line.split('\t')
                data_row[0] = binary_one_hot(data_row[0])  # Convert the label to one-hot
                data_row[1] = ast.literal_eval(data_row[1])  # Convert the logits to a float list
                data_row[2] = int(data_row[2])  # Convert the query id to an integer

                doc_id_index = len(self.doc_id_map)  # Convert the document id to an integer
                self.doc_id_map[doc_id_index] = data_row[3]
                data_row[3] = doc_id_index

                example = Example.fromlist(data_row, self.fields)

                if data_row[2] in dev_splits:
                    self.dev_examples.append(example)
                elif data_row[2] in test_splits:
                    self.test_examples.append(example)
                else:
                    self.train_examples.append(example)

    def get_splits(self, dataset_dir, dev_splits, test_splits, vectors_name, vectors_cache, device, batch_size=64):
        dataset_path = os.path.join(dataset_dir, 'robust04', 'robust04.logits_bert_msmarco_mb.tsv')
        self.load_examples(dataset_path, dev_splits, test_splits)

        train_dataset = Dataset(self.train_examples, self.fields)
        train_dataset.sort_key = lambda example: len(example.text)

        dev_dataset = Dataset(self.dev_examples, self.fields)
        dev_dataset.sort_key = lambda example: len(example.text)

        test_dataset = Dataset(self.test_examples, self.fields)
        test_dataset.sort_key = lambda example: len(example.text)

        vectors = Vectors(name=vectors_name, cache=vectors_cache, unk_init=torch.Tensor.zero_)
        self.query_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=vectors)
        self.text_field.build_vocab(train_dataset, dev_dataset, test_dataset, vectors=vectors)

        return BucketIterator.splits((train_dataset, dev_dataset, test_dataset), batch_size=batch_size,
                                     repeat=False, shuffle=True, sort_within_batch=True, device=device)
