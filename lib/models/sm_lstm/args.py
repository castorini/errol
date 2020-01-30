import os

import models.args


def get_args():
    parser = models.args.get_args()

    parser.add_argument('--dataset', type=str, required=True, choices=['microblog', 'msmarco'])
    parser.add_argument('--mode', type=str, default='static', choices=['rand', 'static', 'non-static'])
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--embed-dim', type=int, default=300)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=512)

    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--weight-drop', type=float, default=0.0)
    parser.add_argument('--epoch-decay', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--save-path', type=str, default=os.path.join(os.pardir, 'model_checkpoints', 'sm_lstm'))
    parser.add_argument('--trained-model', type=str)

    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--distill-div', type=float, default=1)
    parser.add_argument('--distill-mult', type=float, default=4)

    args = parser.parse_args()
    return args


