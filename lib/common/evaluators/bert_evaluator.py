import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from common.evaluators import Evaluator


class BertEvaluator(Evaluator):
    def __init__(self, model, examples, args):
        self.args = args
        self.model = model
        self.examples = examples

    def evaluate(self):
        input_ids = torch.tensor([f.input_ids for f in self.examples], dtype=torch.long)
        token_type_ids = torch.tensor([f.token_type_ids for f in self.examples], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in self.examples], dtype=torch.long)
        label_ids = torch.tensor([f.label for f in self.examples], dtype=torch.long)

        dataset = TensorDataset(input_ids, token_type_ids, attention_mask, label_ids)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)

        self.model.eval()

        total_loss = 0
        target_labels = list()
        predicted_labels = list()

        for input_ids, token_type_ids, attention_mask, label_ids in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = input_ids.to(self.args.device)
            token_type_ids = token_type_ids.to(self.args.device)
            attention_mask = attention_mask.to(self.args.device)
            label_ids = label_ids.to(self.args.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

            predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
            target_labels.extend(torch.argmax(label_ids, dim=1).cpu().detach().numpy())
            loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            total_loss += loss.item()

        scores = self.calculate_scores(predicted_labels, target_labels, total_loss/len(self.examples))
        return scores
