import torch
import torch.nn.functional as F

from common.evaluators import Evaluator


class TorchtextEvaluator(Evaluator):
    def __init__(self, model, dataloader, doc_id_map, args):
        super().__init__(model, args)
        self.dataloader = dataloader
        self.doc_id_map = doc_id_map

    def evaluate(self):
        self.model.eval()
        self.dataloader.init_epoch()
        total_loss = 0

        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            # Temporal averaging
            old_params = self.model.get_params()
            self.model.load_ema_params()

        doc_ids = list()
        query_ids = list()
        predicted_scores = list()

        for batch_idx, batch in enumerate(self.dataloader):
            if hasattr(self.model, 'tar') and self.model.tar:
                logits, rnn_outs = self.model(batch.query[0], batch.input[0], lengths=(batch.query[1], batch.input[1]))
            else:
                logits = self.model(batch.query[0], batch.input[0], lengths=(batch.query[1], batch.input[1]))

            doc_ids.extend((self.doc_id_map[x.item()] for x in batch.doc_id))
            query_ids.extend(batch.query_id.cpu().detach().numpy())
            predicted_scores.extend(logits.cpu().detach().numpy()[:, 0])
            loss = F.cross_entropy(logits, torch.argmax(batch.label, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if hasattr(self.model, 'tar') and self.model.tar:
                # Temporal activation regularization
                loss += (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()
            total_loss += loss.item()

        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            # Temporal averaging
            self.model.load_params(old_params)

        metrics = self.calculate_metrics(query_ids, doc_ids, predicted_scores)
        metrics['loss'] = total_loss / len(self.dataloader.dataset)
        return metrics
