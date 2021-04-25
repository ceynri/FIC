import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from Ganloss import GANloss
from pytorch_msssim import ssim

class GAN(nn.Module):
    def __init__(self, train, lmbda1=1000, lmbda2 = 100):
        super(GAN, self).__init__()
        self.netG = nn.DataParallel(Generator().cuda(), [4, 5])
        if train:
            self.netD = nn.DataParallel(Discriminator().cuda(), [4, 5])
            self.criterionD = GANloss(gan_mode="vanilla")
            self.criterionG = GANloss(gan_mode="vanilla")
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.lmbda1 = lmbda1
            self.lmbda2 = lmbda2

    def forward(self, x):
        self.real = x
        self.x_gen = self.netG(self.real)
        return self.x_gen

    def backward_G(self, total_it, writer):
        pred_fake = self.netD(self.x_gen)
        print(f"G_pred_real: {torch.mean(pred_fake).item():.6f}")
        writer.add_scalar('G_pred_real', torch.mean(pred_fake).item(), total_it)
        self.loss_G_GAN = self.criterionG(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.x_gen, self.real)
        self.ssim_loss = 1 - ssim(self.real, self.x_gen)
        self.loss_G = 10 * self.loss_G_GAN + self.lmbda1 * self.loss_G_L1 + self.lmbda2 * self.ssim_loss
        self.loss_G.backward()

    def backward_D(self, total_it, writer):
        pred_fake = self.netD(self.x_gen.detach())
        self.loss_D_fake = self.criterionD(pred_fake, False)
        pred_real = self.netD(self.real)
        self.loss_D_real = self.criterionD(pred_real, True)
        writer.add_scalar('pred_real', torch.mean(pred_real).item(), total_it)
        writer.add_scalar('pred_fake', torch.mean(pred_fake).item(), total_it)
        print(f"pred_real: {torch.mean(pred_real).item():.6f}"
                f"\tpred_fake: {torch.mean(pred_fake).item():.6f}")
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimize_parameters(self, i, total_it, writer):
        # update D
        if i % 10 == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D(total_it, writer)
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G(total_it, writer)
        self.optimizer_G.step()


