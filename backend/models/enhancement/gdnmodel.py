import torch.nn as nn

from compressai.layers import GDN
from compressai.models import CompressionModel


class Analysis(nn.Module):
    '''
    Analysis net
    '''

    def __init__(self, out_channel_N=128):
        super(Analysis, self).__init__()
        #d = (d - kennel_size + 2 * padding) / stride + 1s
        #batch*128*64*64
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        self.gdn1 = GDN(out_channel_N)
        # batch*128*32*32
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.gdn2 = GDN(out_channel_N)
        # batch*128*16*16
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2, bias=False)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.conv3(x)
        return x


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


class GdnModel(CompressionModel):
    def __init__(self, N=128):
        super().__init__(entropy_bottleneck_channels=N)

        self.encoder = Analysis(out_channel_N=N)
        self.decoder = Synthesis(out_channel_N=N)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, _ = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return x_hat

    def compress(self, x):
        y = self.encoder(x)
        self.entropy_bottleneck.update()
        compressed_data = self.entropy_bottleneck.compress(y)
        return compressed_data, y.size()[-2:]

    def decompress(self, data):
        self.entropy_bottleneck.update()
        bottleneck = self.entropy_bottleneck.decompress(data[0], data[1])
        texture = self.decoder(bottleneck)
        return texture
