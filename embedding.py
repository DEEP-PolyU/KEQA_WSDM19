import torch
from torch import nn
import torch.nn.functional as F

class EmbedVector(nn.Module):
    def __init__(self, config):
        super(EmbedVector, self).__init__()
        self.config = config
        target_size = config.label
        self.embed = nn.Embedding(config.words_num, config.words_dim)
        if config.train_embed == False:
            self.embed.weight.requires_grad = False
        if config.qa_mode.upper() == 'LSTM':
            self.lstm = nn.LSTM(input_size=config.words_dim,
                               hidden_size=config.hidden_size,
                               num_layers=config.num_layer,
                               dropout=config.rnn_dropout,
                               bidirectional=True)
        elif config.qa_mode.upper() == 'GRU':
            self.gru = nn.GRU(input_size=config.words_dim,
                                hidden_size=config.hidden_size,
                                num_layers=config.num_layer,
                                dropout=config.rnn_dropout,
                                bidirectional=True)
        self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
        self.nonlinear = nn.Tanh()
        self.attn = nn.Sequential(
            nn.Linear(config.hidden_size * 2 + config.words_dim, config.hidden_size),
            self.nonlinear,
            nn.Linear(config.hidden_size, 1)
        )
        self.hidden2tag = nn.Sequential(
            nn.Linear(config.hidden_size * 2 + config.words_dim, config.hidden_size * 2),
            nn.BatchNorm1d(config.hidden_size * 2),
            self.nonlinear,
            self.dropout,
            nn.Linear(config.hidden_size * 2, target_size)
        )
        if config.qa_mode.upper() == "CNN":
            input_channel = 1
            Ks = 3
            self.conv1 = nn.Conv2d(input_channel, config.output_channel, (2, config.words_dim), padding=(1, 0))
            self.conv2 = nn.Conv2d(input_channel, config.output_channel, (3, config.words_dim), padding=(2, 0))
            self.conv3 = nn.Conv2d(input_channel, config.output_channel, (4, config.words_dim), padding=(3, 0))
            self.dropout = nn.Dropout(config.cnn_dropout)
            self.fc1 = nn.Linear(Ks * config.output_channel, target_size)

    def forward(self, x):
        # x = (sequence length, batch_size, dimension of embedding)
        text = x.text
        x = self.embed(text)
        num_word, batch_size, words_dim = x.size()
        # h0 / c0 = (layer*direction, batch_size, hidden_dim)
        if self.config.qa_mode.upper() == 'CNN':
            x = x.transpose(0, 1).contiguous().unsqueeze(1)  # (batch, channel_input, sent_len, embed_dim)
            x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
            # (batch, channel_output, ~=sent_len) * Ks
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # max-over-time pooling
            # (batch, channel_output) * Ks
            x = torch.cat(x, 1)  # (batch, channel_output * Ks)
            x = self.dropout(x)
            scores = self.fc1(x)  # (batch, target_size)
            scores = nn.functional.normalize(scores, dim=1)
        elif self.config.qa_mode.upper() != 'LSTM' and self.config.qa_mode.upper() != 'GRU':
            print("Wrong Entity Prediction Mode")
            exit(1)
        else:
            if self.config.qa_mode.upper() == 'LSTM':
                outputs, (ht, ct) = self.lstm(x)
            elif self.config.qa_mode.upper() == 'GRU':
                outputs, ht = self.gru(x)
            outputs = outputs.view(-1, outputs.size(2))
            x = x.view(-1, words_dim)
            attn_weights = F.softmax(self.attn(torch.cat((x, outputs), 1)), dim=1)
            attn_applied = torch.bmm(torch.diag(attn_weights[:, 0]).unsqueeze(0), outputs.unsqueeze(0))
            outputs = torch.cat((x, attn_applied.squeeze(0)), 1)
            tags = self.hidden2tag(outputs).view(num_word, batch_size, -1)
            scores = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)
        return scores