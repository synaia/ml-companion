import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True)
        # self.rnn = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, hidden = self.rnn(x)
        out = hidden[-1, :, :] # final hidden state from the last layer
        out = self.fc(out)
        return out

model = RNN(5, 32)
print(model)
rd = torch.randn(4, 1, 5)
print(rd)
# print(model(rd))