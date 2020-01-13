import torch
import torch.nn as nn

import torch.nn.functional as F


class KimCNNEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        dataset = args.dataset
        words_num = args.words_num
        words_dim = args.words_dim
        target_class = args.target_class
        output_channel = args.output_channel

        if args.mode == 'rand':
            input_channel = 1
            rand_embed_init = torch.Tensor(words_num, words_dim).uniform_(-0.25, 0.25)
            self.embed = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)
        elif args.mode == 'static':
            input_channel = 1
            self.static_embed = nn.Embedding.from_pretrained(dataset.fields['input'].vocab.vectors, freeze=True)
        elif args.mode == 'non-static':
            input_channel = 1
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.fields['input'].vocab.vectors, freeze=False)
        elif args.mode == 'multichannel':
            input_channel = 2
            self.static_embed = nn.Embedding.from_pretrained(dataset.fields['input'].vocab.vectors, freeze=True)
            self.non_static_embed = nn.Embedding.from_pretrained(dataset.fields['input'].vocab.vectors, freeze=False)
        else:
            raise ValueError("Unsupported embedding mode")

        self.conv1 = nn.Conv2d(input_channel, output_channel, (3, words_dim), padding=(2,0))
        self.conv2 = nn.Conv2d(input_channel, output_channel, (4, words_dim), padding=(3,0))
        self.conv3 = nn.Conv2d(input_channel, output_channel, (5, words_dim), padding=(4,0))

        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(3 * output_channel, target_class)

    def forward(self, x):
        if self.args.mode == 'rand':
            word_input = self.embed(x)  # (batch, sent_len, embed_dim)
            x = word_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.args.mode == 'static':
            static_input = self.static_embed(x)
            x = static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.args.mode == 'non-static':
            non_static_input = self.non_static_embed(x)
            x = non_static_input.unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
        elif self.args.mode == 'multichannel':
            non_static_input = self.non_static_embed(x)
            static_input = self.static_embed(x)
            x = torch.stack([non_static_input, static_input], dim=1)  # (batch, channel_input=2, sent_len, embed_dim)

        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]  # (batch, channel_output, ~=sent_len) * num_conv

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch, channel_output) * num_conv
        x = torch.cat(x, 1)  # (batch, channel_output * num_conv)
        return x
