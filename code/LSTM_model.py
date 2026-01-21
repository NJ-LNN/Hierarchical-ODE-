import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMEncoder, self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)

    def forward(self, x, time_points):
        # x shape: [Seq_Len, Batch, Input_Dim]

        outputs, (h_n, c_n) = self.lstm(x)

        return h_n[-1, :, :]


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super(LSTMDecoder, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding, time_points):
        seq_len = time_points.shape[0]
        batch_size = embedding.shape[0]


        h_0 = embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)

        c_0 = torch.zeros_like(h_0).to(embedding.device)


        hidden_state = (h_0, c_0)

        decoder_input = torch.zeros(batch_size, self.output_dim).to(embedding.device)
        outputs = []

        for t in range(seq_len):

            decoder_output, hidden_state = self.lstm(decoder_input.unsqueeze(0), hidden_state)

            # decoder_output: [1, Batch, Hidden]
            output_rssi = self.output_proj(decoder_output.squeeze(0))
            outputs.append(output_rssi)
            decoder_input = output_rssi

        return torch.stack(outputs)


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers)
        self.decoder = LSTMDecoder(hidden_dim, input_dim, num_layers)

    def forward(self, x, time_points):
        embedding = self.encoder(x, time_points)
        return self.decoder(embedding, time_points)