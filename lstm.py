import torch
import torch.nn as nn


class MyLstm(nn.Module):
    def __init__(self, input_size=28, hidden_size=128, num_layers=1, num_classes=10):
        super(MyLstm, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, 10)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)  # remove channel dimension if present

        output, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])

        return out
