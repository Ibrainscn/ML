import torch
import torch.nn as nn


device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
class LSTM(nn.Module):
    def __init__(self, input_shape, model_params, num_classes):
        super(LSTM, self).__init__()
        self.data_shape = input_shape
        self.input_size = input_shape[-1]
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params.get("num_layers", 1)
        self.bidirectional = model_params.get("bidirectional", True)
        self.dropout = model_params.get("dropout", 0.2)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        self.num_directions = 2 if self.bidirectional else 1
        self.fc = nn.Linear(self.hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)

        return out