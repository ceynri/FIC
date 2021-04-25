import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.resnet = InceptionResnetV1()

    def forward(self, x):
        return self.resnet(x)