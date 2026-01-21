import torch
import torch.nn as nn
from torchdiffeq import odeint




class ODEFunc(nn.Module):
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim * 2),
            nn.SiLU(),  
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, t, z):

        t_vec = torch.ones(z.shape[0], 1).to(z.device) * t
        tz = torch.cat([z, t_vec], dim=1)
        return self.net(tz)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.ode_func = ODEFunc(hidden_dim)


        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, x, time_points):
        """
        x: (SeqLen, Batch, InputDim)
        time_points: (SeqLen)
        """
        seq_len, batch_size, _ = x.shape


        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)


        for i in range(seq_len):

            if i > 0:
                t_span = time_points[i - 1:i + 1]

                h_evolved = odeint(self.ode_func, h, t_span, method='rk4')
                h = h_evolved[-1]


            current_x = x[i]
            h = self.gru_cell(current_x, h)


        return h


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.ode_func = ODEFunc(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, embedding, time_points):

        z_t_recon = odeint(self.ode_func, embedding, time_points, method='dopri5')

        output = self.output_proj(z_t_recon)
        return output


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, input_dim)

    def forward(self, x, time_points):
        embedding = self.encoder(x, time_points)
        return self.decoder(embedding, time_points)