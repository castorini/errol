import torch
import torch.nn as nn

from models.sm_cnn.encoder import KimCNNEncoder


class SiameseCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = KimCNNEncoder(args)
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(6 * args.output_channel, args.target_class)

    def forward(self, query, input, **kwargs):
        query = self.encoder(query)  # (batch, channel_output * num_conv)
        input = self.encoder(input)  # (batch, channel_output * num_conv)
        x = torch.cat([query, input], 1)   # (batch, 2 * channel_output * num_conv)
        x = self.dropout(x)
        return self.fc1(x)  # (batch, target_size)
