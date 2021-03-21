import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
        self.net_128 = nn.Linear(512, 128)
        self.last_bn = nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.facenet(x)
        x = self.net_128(x)
        return self.last_bn(x)


'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # c: 3 64 128 256 512 512 512 512 512 512
        # s: 256 256 128 64 32 16 8 4 2 1
        self.downconv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.downconv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.downconv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.downconv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.downconv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.downconv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.downconv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True)
        )
        self.downconv8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, True)
        )

        self.downconv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        result = []
        x = self.downconv1(x)
        x = self.downconv2(x)
        result.append(x)
        x = self.downconv3(x)
        result.append(x)
        x = self.downconv4(x)
        result.append(x)
        x = self.downconv5(x)
        result.append(x)
        x = self.downconv6(x)
        result.append(x)
        x = self.downconv7(x)
        result.append(x)
        x = self.downconv8(x)
        result.append(x)
        x = self.downconv9(x)
        result.append(x)

        return result
'''