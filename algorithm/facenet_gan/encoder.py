import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.facenet = InceptionResnetV1(pretrained='vggface2')
        for param in self.facenet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.facenet(x)
        return x
