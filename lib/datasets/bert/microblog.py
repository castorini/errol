import os
import random

from datasets.common import Dataset, Example
from utils.preprocessing import binary_one_hot


class Microblog(Dataset):
    NAME = 'Microblog'
    NUM_CLASSES = 2

    def __init__(self):
        self.train_examples = list()
        self.dev_examples = list()
        self.test_examples = list()

    def _load_examples(self, dataset_dir, tokenizer, max_seq_length, test_split):
        query_ids = set()
        train_dev_examples = list()
        dataset_files = ['trec_mb_2011.tsv', 'trec_mb_2012.tsv', 'trec_mb_2013.tsv', 'trec_mb_2014.tsv']

        for filename in dataset_files:
            with open(os.path.join(dataset_dir, 'microblog', filename), 'r') as tsv_file:
                for line in tsv_file:
                    data_row = line.split('\t')
                    label = binary_one_hot(data_row[0])
                    query_id = data_row[1].strip()
                    doc_id = data_row[2].strip()
                    encoded_input = tokenizer.encode_plus(data_row[3], data_row[4],
                                                          max_length=max_seq_length,
                                                          pad_to_max_length=True)

                    example = Example(input_ids=encoded_input['input_ids'],
                                      token_type_ids=encoded_input['token_type_ids'],
                                      attention_mask=encoded_input['attention_mask'],
                                      label=label, doc_id=doc_id, query_id=query_id)

                    query_ids.add(query_id)
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

    def get_splits(self, dataset_dir, tokenizer, max_seq_length, test_split='2014'):
        self._load_examples(dataset_dir, tokenizer, max_seq_length, test_split)
        return self.train_examples, self.dev_examples, self.test_examples
