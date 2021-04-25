import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from pytorch_msssim import ssim
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np

class GAN(nn.Module):
    def __init__(self, train, cuda_list, lmbda1=10, lmbda2 = 1):
        super(GAN, self).__init__()
        self.netG = nn.DataParallel(Generator().cuda(), cuda_list)
        self.lambda_gp = 10
        if train:
            self.netD = nn.DataParallel(Discriminator().cuda(), cuda_list)
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(), lr=0.0002)
            self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(), lr=0.0002)
            self.lmbda1 = lmbda1
            self.lmbda2 = lmbda2

    def forward(self, x):
        self.real = x
        self.x_gen = self.netG(self.real)
        return self.x_gen

    def backward_G(self):
        pred_fake = self.netD(self.x_gen)
        self.loss_G_GAN = -torch.mean(pred_fake)
        self.loss_G_L1 = self.criterionL1(self.x_gen, self.real)
        self.ssim_loss = 1 - ssim(self.real, self.x_gen)
        self.loss_G = self.loss_G_GAN + self.lmbda1 * self.loss_G_L1 + self.lmbda2 * self.ssim_loss
        self.loss_G.backward()

    def backward_D(self):
        pred_fake = self.netD(self.x_gen.detach())
        pred_real = self.netD(self.real)
        gradient_penalty = self.compute_gradient_penalty(self.netD, self.real.data, self.x_gen.data)
        self.loss_D = -torch.mean(pred_real) + torch.mean(pred_fake) + self.lambda_gp * gradient_penalty       
        self.loss_D.backward()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        Tensor = torch.cuda.FloatTensor
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).cuda(2)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True).cuda(2)
        d_interpolates = D(interpolates)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(d_interpolates.size()).cuda(2),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def optimize_parameters(self, i):
        # update D
        if i % 5 == 0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
