import torch
import torch.nn as nn

import torch.nn.functional as F

from models.sm_lstm.weight_drop import WeightDrop


class RegLSTMEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.lstm = nn.LSTM(args.embed_dim, args.hidden_dim, num_layers=args.num_layers,
                            bidirectional=args.bidirectional, batch_first=True)

        if args.weight_drop:
            self.lstm = WeightDrop(self.lstm, ['weight_hh_l0'], dropout=args.weight_drop)

        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        rnn_outs, _ = self.lstm(x)
        rnn_outs_temp = rnn_outs

        if lengths is not None:
            rnn_outs,_ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs, batch_first=True)
            rnn_outs_temp, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outs_temp, batch_first=True)

        x = F.relu(torch.transpose(rnn_outs_temp, 1, 2))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        x = self.dropout(x)
        return x
