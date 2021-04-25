import torch.nn as nn
from compressai.layers import GDN


class Analysis(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, out_channel_N=128):
        super(Analysis, self).__init__()
        #d = (d - kennel_size + 2 * padding) / stride + 1s
        #batch*128*64*64
        self.conv1 = nn.Conv2d(3, out_channel_N, 9, stride=4, padding=4)
        self.gdn1 = GDN(out_channel_N)
        # batch*128*32*32
        self.conv2 = nn.Conv2d(
            out_channel_N, out_channel_N, 5, stride=2, padding=2
        )
        self.gdn2 = GDN(out_channel_N)
        # batch*128*16*16
        self.conv3 = nn.Conv2d(
            out_channel_N, out_channel_N, 5, stride=2, padding=2, bias=False
        )

    def forward(self, x):
        x = self.gdn1(self.conv1(x))
        x = self.gdn2(self.conv2(x))
        x = self.conv3(x)
        return x
