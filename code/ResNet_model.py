import torch
import torch.nn as nn
import torch.nn.functional as F

# --- (ResBlock) ---
class ResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock1d, self).__init__()
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        out = self.norm1(x)
        out = self.relu(out)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += shortcut
        return out

# --- Encoder ---
class ResNetEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResNetEncoder, self).__init__()
        self.in_channels = 16
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_dim, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer1 = self._make_layer(16, 2)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(hidden_dim, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

    def _make_layer(self, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(ResBlock1d(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResBlock1d(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, time_points=None):
        # time_points argument is kept for compatibility with other models (GRU/LSTM)
        # but ResNet usually doesn't use it explicitly in the Encoder
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        embedding = self.flatten(x)
        return embedding

# --- Decoder ---
class ResNetDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(ResNetDecoder, self).__init__()
        self.hidden_dim = hidden_dim

        
        
        self.start_len = 4
        decoder_start_channels = hidden_dim
        self.linear = nn.Linear(hidden_dim, decoder_start_channels * self.start_len)
        self._initial_decoder_channels = decoder_start_channels

       
        self.in_channels = decoder_start_channels
        self.layer1 = self._make_decode_layer(32)
        self.layer2 = self._make_decode_layer(16)
        self.layer3 = self._make_decode_layer(16)

       
        self.final_conv = nn.Sequential(
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_channels, output_dim, kernel_size=3, padding=1)
        )

    def _make_decode_layer(self, out_channels):
        
        layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            ResBlock1d(self.in_channels, out_channels, downsample=nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, 1, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            ))
        )
        self.in_channels = out_channels
        return layer

    def forward(self, embedding, time_points):
        
        seq_len = time_points.shape[0]

        
        x = self.linear(embedding)
        x = x.view(embedding.shape[0], self._initial_decoder_channels, self.start_len)

       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

      
        x = self.final_conv(x)

        
        if x.shape[2] != seq_len:
            x = F.interpolate(x, size=seq_len, mode='linear', align_corners=False)

        return x

class ResNetAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResNetAutoencoder, self).__init__()
        self.encoder = ResNetEncoder(input_dim, hidden_dim)
        self.decoder = ResNetDecoder(hidden_dim, input_dim)

    def forward(self, x, time_points):
        embedding = self.encoder(x, time_points)
        reconstruction = self.decoder(embedding, time_points)
        return reconstruction