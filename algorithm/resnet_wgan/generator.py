import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
import torch

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = torch.squeeze(bottleneck, 1)
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)
        x = self.decoder(x)
        return bottleneck, x