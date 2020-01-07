import abc
import datetime
import os
import shlex
import subprocess

from common.constants import TREC_EVAL_PATH, QREL_PATH_MAP
from utils.io import save_ranks


class Evaluator(abc.ABC):
    def __init__(self, model, args):
        self.args = args
        self.model = model

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError()

    def calculate_metrics(self, query_ids, doc_ids, scores):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(self.args.save_path, self.args.dataset, '%s.txt' % timestamp)
        save_ranks(output_path, query_ids, doc_ids, scores)

        cmd = '%s %s %s -m map -m P.30 -m recip_rank' % (TREC_EVAL_PATH, QREL_PATH_MAP[self.args.dataset], output_path)
        pargs = shlex.split(cmd)
        p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pout, perr = p.communicate()
        pout = [line.split() for line in pout.split(b'\n') if line.strip()]

        metrics = dict()
        for metric, _, value in pout:
            metrics[metric.decode("utf-8").lower()] = float(value)

        return metrics
