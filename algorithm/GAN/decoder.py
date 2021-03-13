import torch.nn as nn
import torch

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