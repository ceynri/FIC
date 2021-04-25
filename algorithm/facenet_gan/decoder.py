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


'''
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # c: 512 1024 1024 1024 1024 1024 512 256 64 3
        # s: 1 2 4 8 16 32 64 128 256 256
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True)
        )
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True)
        )
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True)
        )
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True)
        )
        self.upconv5 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.ReLU(True)
        )
        self.upconv6 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.ReLU(True)
        )
        self.upconv7 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(True)
        )
        self.upconv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64, affine=True),
            nn.ReLU(True)
        )
        self.upconv9 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, result):
        x = result[-1]
        x = self.upconv1(x)
        x = torch.cat((x, result[-2]), 1)
        x = self.upconv2(x)
        x = torch.cat((x, result[-3]), 1)
        x = self.upconv3(x)
        x = torch.cat((x, result[-4]), 1)
        x = self.upconv4(x)
        x = torch.cat((x, result[-5]), 1)
        x = self.upconv5(x)
        x = torch.cat((x, result[-6]), 1)
        x = self.upconv6(x)
        x = torch.cat((x, result[-7]), 1)
        x = self.upconv7(x)
        x = torch.cat((x, result[-8]), 1)
        x = self.upconv8(x)
        x = self.upconv9(x)

        return x
'''