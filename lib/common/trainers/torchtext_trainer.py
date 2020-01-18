import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.trainers import Trainer


class TorchtextTrainer(Trainer):
    def __init__(self, model, dataloader, optimizer, evaluator, args):
        super().__init__(model, optimizer, evaluator, args)
        self.dataloader = dataloader

    def train_epoch(self):
        self.dataloader.init_epoch()
        for batch in tqdm(self.dataloader, desc="Training"):
            self.iterations += 1
            self.model.train()
            self.optimizer.zero_grad()

            if hasattr(self.model, 'tar') and self.model.tar:
                logits, rnn_outs = self.model(batch.query[0], batch.input[0], lengths=(batch.query[1], batch.input[1]))
            else:
                logits = self.model(batch.query[0], batch.input[0], lengths=(batch.query[1], batch.input[1]))

            loss = F.cross_entropy(logits, torch.argmax(batch.label.data, dim=1))

            if self.args.distill:
                loss += self.args.distill_mult * F.kl_div(F.log_softmax(logits), F.softmax(batch.logits / self.args.distill_div))

            if hasattr(self.model, 'tar') and self.model.tar:
                loss += self.model.tar * (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()
            if hasattr(self.model, 'ar') and self.model.ar:
                loss += self.model.ar * (rnn_outs[:]).pow(2).mean()

            loss.backward()
            self.optimizer.step()

            if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
                # Temporal averaging
                self.model.update_ema()
