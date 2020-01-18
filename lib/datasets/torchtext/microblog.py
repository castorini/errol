import ast
import csv
import os
import random
import sys
from collections import defaultdict

from torchtext.data import Example

from common.constants import DATASET_DIR
from datasets.torchtext.common import TorchtextDataset
from utils.preprocessing import binary_one_hot

# Set upper limit on parsed CSV fields
csv.field_size_limit(sys.maxsize)


class Microblog(TorchtextDataset):
    NAME = 'TREC Microblog'
    NUM_CLASSES = 2

    def __init__(self, test_split='2014'):
        super().__init__()
        self._load_examples(test_split)

    def _load_examples(self, test_split):
        logits = defaultdict(dict)
        query_ids = set()
        train_dev_examples = list()
        dataset_files = ['trec_mb_2011.tsv', 'trec_mb_2012.tsv', 'trec_mb_2013.tsv', 'trec_mb_2014.tsv']

        with open(os.path.join(DATASET_DIR, 'microblog', 'trec_mb_logits_bert_base.tsv'), 'r') as tsv_file:
            for line in tsv_file:
                l, query_id, doc_id = line.strip().split('\t')
                logits[query_id][doc_id] = l

        for filename in dataset_files:
            with open(os.path.join(DATASET_DIR, 'microblog', filename), 'r') as tsv_file:
                for line in tsv_file:
                    line_split = line.split('\t')
                    data_row = list()
                    data_row.append(binary_one_hot(line_split[0]))  # Convert the label to one-hot

                    query_id_string = line_split[1].strip()
                    doc_id_string = line_split[2].strip()
                    data_row.append(ast.literal_eval(logits[query_id_string][doc_id_string]))

                    query_id = int(query_id_string)  # Convert the query id to an integer
                    query_ids.add(query_id)
                    data_row.append(query_id)

                    doc_id_index = len(self.doc_id_map)  # Convert the document id to an integer
                    self.doc_id_map[doc_id_index] = doc_id_string
                    data_row.append(doc_id_index)

                    data_row.append(line_split[3])  # Add the query text field
                    data_row.append(line_split[4])  # Add the document text field

                    example = Example.fromlist(data_row, self.fields)

                    if test_split in filename:
                        self.test_examples.append(example)
                    else:
                        train_dev_examples.append(example)

        dev_queries = set(random.sample(query_ids, k=round(0.15 * len(query_ids))))
        for example in train_dev_examples:
            if example.query_id in dev_queries:
                self.dev_examples.append(example)
            else:
                self.train_examples.append(example)
