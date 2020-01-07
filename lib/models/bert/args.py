import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['microblog'])
    parser.add_argument('--save-path', type=str, default=os.path.join(os.pardir, 'model_checkpoints', 'bert'))
    parser.add_argument('--cache-dir', default='cache', type=str)
    parser.add_argument('--trained-model', default=None, type=str)
    parser.add_argument('--max-seq-length', default=128, type=int,
                        help='The maximum total input sequence length after WordPiece tokenization')
    parser.add_argument('--warmup-proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup for')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass')

    args = parser.parse_args()
    return args
