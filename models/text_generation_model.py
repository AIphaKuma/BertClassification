import torch
import torch.nn as nn

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, rnn_type='LSTM'):
        super(TextGenerationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn_type = rnn_type.upper()
        if self.rnn_type == 'RNN':
            self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("Type de RNN inconnu.")
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = out.reshape(-1, out.size(2))
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size, device):
        num_layers = self.rnn.num_layers
        hidden_size = self.rnn.hidden_size
        if self.rnn_type == 'LSTM':
            return (torch.zeros(num_layers, batch_size, hidden_size, device=device),
                    torch.zeros(num_layers, batch_size, hidden_size, device=device))
        else:
            return torch.zeros(num_layers, batch_size, hidden_size, device=device)