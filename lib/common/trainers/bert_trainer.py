import datetime
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm
from tqdm import trange

from common.trainers import Trainer


class BertTrainer(Trainer):
    def __init__(self, model, examples, optimizer, scheduler, evaluator, args):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.examples = examples

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.snapshot_path = os.path.join(self.args.save_path, self.args.dataset, '%s.pt' % timestamp)
        self.num_train_optimization_steps = int(len(self.examples) / args.batch_size /
                                                args.gradient_accumulation_steps) * args.epochs

        self.log_header = 'Epoch Iteration Progress   Dev/P_30  Dev/MAP  Dev/MRR   Dev/Loss'
        self.log_template = ' '.join('{:>5.0f},{:>9.0f},{:>6.0f}/{:<5.0f} {:>6.4f},{:>8.4f},{:8.4f},{:10.4f}'.split(','))

        self.iterations = 0
        self.best_dev_f1 = 0
        self.unimproved_iters = 0
        self.early_stop = False

    def train_epoch(self, train_dataloader):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            self.model.train()
            batch = tuple(t.to(self.args.device) for t in batch)
            input_ids, token_type_ids, attention_mask, label_ids = batch
            logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
            loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.iterations += 1

    def train(self):
        print("Number of train examples: ", len(self.examples))
        print("Number of steps:", self.num_train_optimization_steps)

        input_ids = torch.tensor([f.input_ids for f in self.examples], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in self.examples], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in self.examples], dtype=torch.long)
        label_ids = torch.tensor([f.label for f in self.examples], dtype=torch.long)

        dataset = TensorDataset(input_ids, token_type_ids, attention_mask, label_ids)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)

        for epoch in range(int(self.args.epochs)):
            self.train_epoch(dataloader)
            dev_scores = self.evaluator.evaluate()

            # Print validation results
            tqdm.write(self.log_header)
            tqdm.write(self.log_template.format(epoch + 1, self.iterations, epoch + 1, self.args.epochs,
                                                dev_scores['p_30'], dev_scores['map'], dev_scores['recip_rank'],
                                                dev_scores['loss']))

            # Update validation results
            if dev_scores['map'] > self.best_dev_f1:
                self.unimproved_iters = 0
                self.best_dev_f1 = dev_scores['map']
                torch.save(self.model, self.snapshot_path)
            else:
                self.unimproved_iters += 1
                if self.unimproved_iters >= self.args.patience:
                    self.early_stop = True
                    break
