import abc
import datetime
import os

import torch

from common.constants import DEV_LOG_HEADER, DEV_LOG_TEMPLATE


class Trainer(abc.ABC):
    def __init__(self, model, optimizer, evaluator, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator

        self.iterations = 0
        self.best_dev_map = 0
        self.unimproved_iterations = 0
        self.early_stop = False

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.args.save_path, self.args.dataset, '%s.pt' % timestamp)

    @abc.abstractmethod
    def train_epoch(self):
        raise NotImplementedError()

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch()
            dev_scores = self.evaluator.evaluate()

            # Print validation results
            print(DEV_LOG_HEADER)
            print(DEV_LOG_TEMPLATE.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs, dev_scores['p_30'],
                                          dev_scores['map'], dev_scores['recip_rank'], dev_scores['loss']))

            # Update validation results
            if dev_scores['map'] > self.best_dev_map:
                self.unimproved_iterations = 0
                self.best_dev_map = dev_scores['map']
                torch.save(self.model, self.snapshot_path)
            else:
                self.unimproved_iterations += 1
                if self.unimproved_iterations >= self.args.patience:
                    self.early_stop = True
                    break
