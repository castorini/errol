import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm

from common.trainers import Trainer


class BertTrainer(Trainer):
    def __init__(self, model, examples, optimizer, scheduler, evaluator, args):
        super().__init__(model, optimizer, evaluator, args)
        self.examples = examples
        self.scheduler = scheduler

        input_ids = torch.tensor([f.input_ids for f in self.examples], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in self.examples], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in self.examples], dtype=torch.long)
        label_ids = torch.tensor([f.label for f in self.examples], dtype=torch.long)

        dataset = TensorDataset(input_ids, token_type_ids, attention_mask, label_ids)
        sampler = RandomSampler(dataset)
        self.dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)
        self.num_train_optimization_steps = int(len(self.examples) / args.batch_size /
                                                args.gradient_accumulation_steps) * args.epochs

    def train_epoch(self):
        for step, batch in enumerate(tqdm(self.dataloader, desc="Training")):
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
