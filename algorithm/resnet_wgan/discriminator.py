import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # c: 3 64 128 256 512 512 512 64 1
        # s: 256 128 64 32 16 8 5 2 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=4, stride=1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        return x