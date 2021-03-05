from analysis import Analysis
from synthesis import Synthesis
from compressai.models import CompressionModel


class AutoEncoder(CompressionModel):
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
