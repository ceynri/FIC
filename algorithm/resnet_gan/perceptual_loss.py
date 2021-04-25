import torchvision
import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1

from torchsummary import summary

class Perc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet(
            requires_grad=False
        )
        self.criterion = nn.MSELoss()

    def forward(self, bottleneck, img):
        facenet = self.model(img)
        loss = self.criterion(bottleneck, facenet)
        return loss


class resnet(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        if requires_grad == False:
            for param in self.facenet.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.facenet(x)
        return x

if __name__ == '__main__':
    net = resnet().cuda()