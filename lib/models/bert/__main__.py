import random

import numpy as np
import torch
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

from common.constants import *
from common.evaluators import BertEvaluator
from common.trainers import BertTrainer
from datasets.bert import Microblog
from models.bert.args import get_args


def evaluate_split(model, examples, args, split):
    evaluator = BertEvaluator(model, examples, args)
    scores = evaluator.evaluate()
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), scores['p_30'], scores['map'], scores['recip_rank'], scores['loss']))


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    print('Device:', str(device).upper())
    print('Number of GPUs:', n_gpu)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {
        'microblog': Microblog
    }

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    args.batch_size = args.batch_size // args.gradient_accumulation_steps
    args.device = device
    args.n_gpu = n_gpu
    args.num_labels = dataset_map[args.dataset].NUM_CLASSES

    if not args.trained_model:
        save_path = os.path.join(args.save_path, args.dataset)
        os.makedirs(save_path, exist_ok=True)

    processor = dataset_map[args.dataset]()
    pretrained_vocab_path = PRETRAINED_VOCAB_ARCHIVE_MAP[args.model]
    tokenizer = BertTokenizer.from_pretrained(pretrained_vocab_path)
    train_examples, dev_examples, test_examples = processor.get_splits(args.data_dir, tokenizer, args.max_seq_length)

    print('Dataset:', args.dataset)
    print('No. of train examples:', len(train_examples))
    print('No. of dev examples:', len(dev_examples))
    print('No. of test examples:', len(test_examples))

    pretrained_model_path = args.model if os.path.isfile(args.model) else PRETRAINED_MODEL_ARCHIVE_MAP[args.model]
    model = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=args.num_labels)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=0.01, correct_bias=False)
    num_train_opt_steps = int(len(train_examples) / args.batch_size / args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_training_steps=num_train_opt_steps,
                                                num_warmup_steps=args.warmup_proportion * num_train_opt_steps)

    dev_evaluator = BertEvaluator(model, dev_examples, args)
    trainer = BertTrainer(model, train_examples, optimizer, scheduler, dev_evaluator, args)

    if not args.trained_model:
        trainer.train()
        model = torch.load(trainer.snapshot_path)

    else:
        model = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=args.num_labels)
        model_ = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
        state = {}
        for key in model_.state_dict().keys():
            new_key = key.replace("module.", "")
            state[new_key] = model_.state_dict()[key]
        model.load_state_dict(state)
        model = model.to(device)

    evaluate_split(model, dev_examples, args, split='dev')
    evaluate_split(model, test_examples, args, split='test')
