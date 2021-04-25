import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.DconvOp1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=8, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )
        self.DconvOp2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.DconvOp3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True)
        )
        self.DconvOp4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )
        self.DconvOp5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        self.DconvOp6 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True)
        )
        self.DconvOp7 = nn.Sequential(
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.DconvOp1(x)
        x = self.DconvOp2(x)
        x = self.DconvOp3(x)
        x = self.DconvOp4(x)
        x = self.DconvOp5(x)
        x = self.DconvOp6(x)
        x = self.DconvOp7(x)
        return x