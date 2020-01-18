import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--dataset', type=str, required=True, choices=['microblog', 'msmarco'])
    parser.add_argument('--mode', type=str, default='static', choices=['rand', 'static', 'non-static', 'multichannel'])
    parser.add_argument('--output-channel', type=int, default=100)
    parser.add_argument('--words-dim', type=int, default=300)
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--save-path', type=str, default=os.path.join(os.pardir, 'model_checkpoints', 'sm_cnn'))
    parser.add_argument('--trained-model', type=str)

    args = parser.parse_args()
    return args


