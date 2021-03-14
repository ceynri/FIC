import torch
import torch.nn as nn
from pytorch_msssim import ssim

from .discriminator import Discriminator
from .Ganloss import GANloss
from .generator import Generator


class GAN(nn.Module):
    def __init__(self, train, lmbda1=100, lmbda2=100):
        super(GAN, self).__init__()
        self.netG = nn.DataParallel(Generator().cuda())
        if train:
            self.netD = nn.DataParallel(Discriminator().cuda())
            self.criterionD = GANloss(gan_mode="vanilla")
            self.criterionG = GANloss(gan_mode="vanilla")
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999)
            )
            self.lmbda1 = lmbda1
            self.lmbda2 = lmbda2

    def forward(self, x):
        self.real = x
        self.x_gen = self.netG(self.real)
        return self.x_gen

    def backward_G(self):
        pred_fake = self.netD(self.x_gen)
        self.loss_G_GAN = self.criterionG(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.x_gen, self.real)
        self.ssim_loss = 1 - ssim(self.real, self.x_gen)
        self.loss_G = self.loss_G_GAN + self.lmbda1 * self.loss_G_L1 + self.lmbda2 * self.ssim_loss
        self.loss_G.backward()

    def backward_D(self):
        pred_fake = self.netD(self.x_gen.detach())
        self.loss_D_fake = self.criterionD(pred_fake, False)
        pred_real = self.netD(self.real)
        self.loss_D_real = self.criterionD(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self):
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
