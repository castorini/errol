import os
import random
from copy import deepcopy

import numpy as np
import torch

from common.constants import LOG_HEADER, LOG_TEMPLATE, RETRIEVAL_PATH_MAP, QREL_PATH_MAP
from common.evaluators import TorchtextEvaluator
from common.trainers import TorchtextTrainer
from datasets.torchtext import Microblog
from datasets.torchtext.msmarco import MSMarco
from models.sm_lstm.args import get_args
from models.sm_lstm.model import SiameseLSTM
from utils.rerank import rerank


def evaluate_split(model, dataloader, doc_id_map, split):
    evaluator = TorchtextEvaluator(model, dataloader, doc_id_map, args)
    scores = evaluator.evaluate()
    print('\n' + LOG_HEADER)
    print(LOG_TEMPLATE.format(split.upper(), scores['p_30'], scores['map'], scores['recip_rank'], scores['loss']))
    return evaluator.output_path


if __name__ == '__main__':
    # Set default configuration in args.py
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    print('Device:', str(args.device).upper())
    print('Number of GPUs:', args.n_gpu)

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    dataset_map = {
        'msmarco': MSMarco,
        'microblog': Microblog
    }

    if args.dataset not in dataset_map:
        raise ValueError('Unrecognized dataset')

    dataset = dataset_map[args.dataset]()
    train_iter, dev_iter, test_iter = dataset.get_splits(device=args.device, batch_size=args.batch_size)

    config = deepcopy(args)
    config.dataset = train_iter.dataset
    config.target_class = dataset.NUM_CLASSES
    config.input_vocab_len = len(train_iter.dataset.fields['input'].vocab)
    config.query_vocab_len = len(train_iter.dataset.fields['query'].vocab)

    print('Dataset:', args.dataset)
    print('No. of train examples:', len(train_iter.dataset))
    print('No. of dev examples:', len(dev_iter.dataset))
    print('No. of test examples:', len(test_iter.dataset))

    model = SiameseLSTM(config)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if not args.trained_model:
        save_path = os.path.join(args.save_path, args.dataset)
        os.makedirs(save_path, exist_ok=True)

    parameter = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameter, lr=args.lr, weight_decay=args.weight_decay)

    evaluator = TorchtextEvaluator(model, dev_iter, dataset.doc_id_map, args)
    trainer = TorchtextTrainer(model, train_iter, optimizer, evaluator, args)

    if not args.trained_model:
        trainer.train()
        model.load_state_dict(torch.load(trainer.snapshot_path))
    else:
        model = torch.load(args.trained_model, map_location=lambda storage, location: storage.to(args.device))

    # Calculate dev and test metrics
    dev_ranks_path = evaluate_split(model, dev_iter, dataset.doc_id_map, 'dev')
    test_ranks_path = evaluate_split(model, test_iter, dataset.doc_id_map, 'test')
    rerank(QREL_PATH_MAP[args.dataset], RETRIEVAL_PATH_MAP[args.dataset], dev_ranks_path, test_ranks_path)
