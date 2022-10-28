import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, lstm=False):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = lstm
        if lstm:
            self.rnn = nn.LSTM(input_size, hidden_size,
                               num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size,
                              num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, t):
        x = torch.cat([x, t], dim=-1)
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        if self.lstm:
            c0 = torch.zeros(self.num_layers, x.size(0),
                             self.hidden_size).to(device)
            out, (hn, cn) = self.rnn(x, (h0, c0))
            hn = hn.view(-1, self.hidden_size)
            out = nn.Tanh()(hn)
        else:
            out, _ = self.rnn(x, h0)
        out = nn.Tanh()(self.fc1(out))
        out = self.fc2(out[:, -1, :])
        return out

class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, lstm=False):
        super(Transformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = lstm
        n_head = 2
        head_dim = 16
        dmodel = n_head * head_dim
        self.emb = nn.Linear(input_size, dmodel)
        encoder_layer = nn.TransformerEncoderLayer(dmodel, n_head, dim_feedforward=hidden_size, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(dmodel, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, t):
        x = torch.cat([x, t], dim=-1)
        x = self.emb(x)
        x = self.transformer_encoder(x)
        x = nn.Tanh()(self.fc1(x))
        out = self.fc2(x[:, -1, :])
        return out