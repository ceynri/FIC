import torch
import torch.nn as nn
import math

class RateDistortionLoss(nn.Module):
    '''Custom rate distortion loss with a Lagrangian parameter.'''

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target, x_tex, x_rec):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out['bpp_loss'] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output['likelihoods'].values()
        )
        out['mse_loss'] = self.mse(x_tex, x_rec)
        out['loss'] = self.lmbda * out['mse_loss'] + out['bpp_loss']

        return out
