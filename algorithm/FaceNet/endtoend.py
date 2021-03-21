from analysis import Analysis
from synthesis import Synthesis
from compressai.models import CompressionModel

class AutoEncoder(CompressionModel):
    def __init__(self, N=128):
        super(AutoEncoder, self).__init__(entropy_bottleneck_channels=N)

        self.encoder = Analysis(out_channel_N = N)
        self.decoder = Synthesis(out_channel_N = N)

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.decoder(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }