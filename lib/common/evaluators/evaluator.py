import abc
import datetime
import os

from common.constants import QREL_PATH_MAP
from utils.io import save_ranks, run_trec_eval


class Evaluator(abc.ABC):
    def __init__(self, model, args):
        self.args = args
        self.model = model

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_path = os.path.join(self.args.save_path, self.args.dataset, '%s.txt' % timestamp)

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError()

    def calculate_metrics(self, query_ids, doc_ids, scores):
        save_ranks(self.output_path, query_ids, doc_ids, scores)
        return run_trec_eval(QREL_PATH_MAP[self.args.dataset], self.output_path)

