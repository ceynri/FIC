from analysis import Analysis
from synthesis import Synthesis
from compressai.models import CompressionModel

class AutoEncoder(CompressionModel):
    def __init__(self, N=128):
        super().__init__(entropy_bottleneck_channels=N)

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

'''
class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=128, training=True):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis(out_channel_N=out_channel_N)
        self.Decoder = Synthesis(out_channel_N=out_channel_N)
        self.out_channel_N = out_channel_N
        self.bitEstimator = BitEstimator(channel=out_channel_N)
        #self.training = True
        self.training = training

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()      
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        feature = self.Encoder(input_image)
        # batch_size = feature.size()[0]
        feature_renorm = feature

        # quantization
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)



        # def iclr18_estimate_bits_z(z):
        #     prob = self.bitEstimator(z + 0.5) - self.bitEstimator(z - 0.5)
        #     total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
        #     return total_bits, prob
        
        # total_bits_feature, _ = iclr18_estimate_bits_z(compressed_feature_renorm)
        # im_shape = input_image.size()
        # bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])                     
        
        return recon_image# , bpp_feature
'''
