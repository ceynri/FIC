import torch.nn as nn
from compressai.layers import GDN


class Synthesis(nn.Module):
    '''
    Decode synthesis
    '''

    def __init__(self, out_channel_N=128):
        super(Synthesis, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, 3, 9, stride=4, padding=4, output_padding=3)

    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.deconv3(x)
        return x
