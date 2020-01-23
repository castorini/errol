import torch
import torch.nn as nn

import torch.nn.functional as F


class KimCNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        input_channel = 1

        self.conv1 = nn.Conv2d(input_channel, args.output_channel, (3, args.embed_dim), padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channel, args.output_channel, (4, args.embed_dim), padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channel, args.output_channel, (5, args.embed_dim), padding=(4, 0))
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]  # (batch, channel_output, ~=sent_len) * num_conv

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch, channel_output) * num_conv
        x = torch.cat(x, 1)  # (batch, channel_output * num_conv)
        x = self.dropout(x)
        return x
