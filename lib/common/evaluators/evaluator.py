import abc
import warnings

import numpy as np
from sklearn import metrics


class Evaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError()

    def calculate_scores(self, predicted_labels, target_labels, loss):
        scores = dict()
        predicted_labels = np.array(predicted_labels)
        target_labels = np.array(target_labels)

        with warnings.catch_warnings():
            # Suppress undefined metric warnings from sklearn.metrics
            warnings.simplefilter("ignore")
            scores['loss'] = loss
            scores['accuracy'] = metrics.accuracy_score(target_labels, predicted_labels)
            scores['precision'] = metrics.precision_score(target_labels, predicted_labels, average=None)[0]
            scores['recall'] = metrics.recall_score(target_labels, predicted_labels, average=None)[0]
            scores['f1'] = metrics.f1_score(target_labels, predicted_labels, average=None)[0]

        return scores
