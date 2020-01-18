import ast
import csv
import os
import sys

from torchtext.data import Example

from common.constants import DATASET_DIR
from datasets.torchtext.common import TorchtextDataset
from utils.preprocessing import binary_one_hot

# Set upper limit on parsed CSV fields
csv.field_size_limit(sys.maxsize)


class MSMarco(TorchtextDataset):
    NAME = 'MS MARCO'
    NUM_CLASSES = 2

    def __init__(self):
        super().__init__()
        self.collection = dict()
        self._load_examples()

    def _load_queries(self, file_path):
        queries = dict()
        with open(file_path, 'r') as tsv_file:
            for line in tsv_file:
                query_id, query = line.strip().split('\t')
                queries[query_id] = query
        return queries

    def _create_example(self, line, queries):
        label, logits, query_id, doc_id = line.strip().split('\t')
        data_row = list()
        data_row.append(binary_one_hot(label))
        data_row.append(ast.literal_eval(logits))
        data_row.append(int(query_id))

        doc_id_index = len(self.doc_id_map)  # Convert the document id to an integer
        self.doc_id_map[doc_id_index] = doc_id
        data_row.append(doc_id_index)

        data_row.append(queries[query_id])  # Add the query text field
        data_row.append(self.collection[doc_id])  # Add the document text field

        return Example.fromlist(data_row, self.fields)

    def _load_examples(self):
        with open(os.path.join(DATASET_DIR, 'msmarco', 'msmarco.collection.tsv'), 'r') as tsv_file:
            for line in tsv_file:
                doc_id, doc_text = line.strip().split('\t')
                self.collection[doc_id] = doc_text

        train_queries = self._load_queries(os.path.join(DATASET_DIR, 'msmarco', 'msmarco.train.queries.tsv'))
        with open(os.path.join(DATASET_DIR, 'msmarco', 'msmarco.train.logits_bert_large.tsv'), 'r') as tsv_file:
            for line in tsv_file:
                self.train_examples.append(self._create_example(line, train_queries))

        dev_queries = self._load_queries(os.path.join(DATASET_DIR, 'msmarco', 'msmarco.dev.queries.small.tsv'))
        with open(os.path.join(DATASET_DIR, 'msmarco', 'msmarco.dev.logits_bert_large.tsv'), 'r') as tsv_file:
            for line in tsv_file:
                self.dev_examples.append(self._create_example(line, dev_queries))
