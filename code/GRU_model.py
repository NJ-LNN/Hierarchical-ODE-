import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)

    def forward(self, x, time_points):
        # x shape: [Seq_Len, Batch, Input_Dim]
        outputs, h_n = self.gru(x)
        return h_n[-1, :, :]

class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers=2):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_size=output_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding, time_points):
        seq_len = time_points.shape[0]
        batch_size = embedding.shape[0]
        h_0 = embedding.unsqueeze(0).repeat(self.num_layers, 1, 1)
        hidden_state = h_0
        decoder_input = torch.zeros(batch_size, self.output_dim).to(embedding.device)
        outputs = []
        for t in range(seq_len):
            decoder_output, hidden_state = self.gru(decoder_input.unsqueeze(0), hidden_state)
            output_rssi = self.output_proj(decoder_output.squeeze(0))
            outputs.append(output_rssi)
            decoder_input = output_rssi
        return torch.stack(outputs)

class GRUAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super(GRUAutoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(hidden_dim, input_dim, num_layers)

    def forward(self, x, time_points):
        embedding = self.encoder(x, time_points)
        return self.decoder(embedding, time_points)