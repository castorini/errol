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


class Robust04(TorchtextDataset):
    NAME = 'Robust04'
    NUM_CLASSES = 2

    def __init__(self, dev_splits, test_splits):
        super().__init__()
        self._load_examples(dev_splits, test_splits)

    def _load_examples(self, dev_splits, test_splits):
        dataset_path = os.path.join(DATASET_DIR, 'robust04', 'robust04.logits_bert_msmarco_mb.tsv')
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
