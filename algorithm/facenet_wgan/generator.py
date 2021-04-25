import torch.nn as nn
import torch
from encoder import Encoder
from decoder import Decoder

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = torch.squeeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = torch.unsqueeze(x, 3)
        x = self.decoder(x)

        return x