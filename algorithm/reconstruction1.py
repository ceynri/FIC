from torch import nn
import torch
# It is worth mentioning here that 
# the DconvOP is composed of a series procedures, 
# including de-convolution layer, batchnormalization layer
# and ReLU activation (the activation function of the last layer is Tanh).

class Dconv(nn.Module):
    def __init__(self):
        super(Dconv,self).__init__()
        # DconvOp
        self.DconvOp1 = nn.Sequential(
            nn.ConvTranspose2d(512,512,kernel_size=8,stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.DconvOp2 = nn.Sequential(
            nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.DconvOp3 = nn.Sequential(
            nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.DconvOp4 = nn.Sequential(
            nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )        
        self.DconvOp5 = nn.Sequential(
            nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.DconvOp6 = nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.DconvOp7 = nn.Sequential(
            nn.ConvTranspose2d(16,3,kernel_size=3,stride=1,padding=1), 
            nn.BatchNorm2d(3),
            nn.Tanh()
            # nn.ReLU(True)
        )       
    def forward(self,x):
        x = self.DconvOp1(x)
        x = self.DconvOp2(x)
        x = self.DconvOp3(x)
        x = self.DconvOp4(x)
        x = self.DconvOp5(x)
        x = self.DconvOp6(x)
        x = self.DconvOp7(x)    
        return x         