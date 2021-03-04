import torch.nn as nn
from gdn import GDN


class Analysis(nn.Module):
    '''
    Analysis net
    '''

    def __init__(self, out_channel_N=128):
        super(Analysis, self).__init__()
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        self.gdn1 = GDN(out_channel_N)
        self.conv2 = nn.Conv2d(
            out_channel_N, out_channel_N, 5, stride=2, padding=2)
        self.gdn2 = GDN(out_channel_N)
        self.conv3 = nn.Conv2d(out_channel_N, out_channel_N,
                               5, stride=2, padding=2, bias=False)

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.conv3(x)
        return x
