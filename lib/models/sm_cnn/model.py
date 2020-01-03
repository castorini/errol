import torch
import torch.nn as nn

from lib.models.sm_cnn.encoder import KimCNNEncoder


class SiameseCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = KimCNNEncoder(config)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(2 * config.output_channel * self.encoder.num_conv, config.target_class)

    def forward(self, query, text):
        query = self.encoder(query)  # (batch, channel_output * num_conv)
        text = self.encoder(text)  # (batch, channel_output * num_conv)
        x = torch.cat([query, text], 1)   # (batch, 2 * channel_output * num_conv)
        x = self.dropout(x)
        return self.fc1(x)  # (batch, target_size)
