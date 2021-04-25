import torch
from torch.utils.data import DataLoader
from  dataset import dataset
from torch.utils.data.dataloader import default_collate
from network import GAN
import cv2
from tensorboardX import SummaryWriter


def collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    while len(batch) < 16:
        j = len(batch)
        for i in (0, j):
            if len(batch) < 16:
                batch.append(batch[i])
    return default_collate(batch)

if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)
    dataset = dataset('/data/chenyangrui/train')
    ds = DataLoader(
        dataset=dataset,
        num_workers=10,
        batch_size=40,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate
    )
    net = GAN(train=True).cuda(2)
    epoch = 0
    total_it = 0
    writer = SummaryWriter('/data/chenyangrui/resnetWGan/tsb')
    load = True
    if load:
        param = torch.load('/data/chenyangrui/resnetWGan/resnetWGan_checkpoints')
        epoch = param['epoch']
        net.load_state_dict(param['model_state_dict'])
        total_it = param['total_it']

    print("start train")
    while(epoch < 100):
        net.train()
        for i, d in enumerate(ds):
            total_it += 1
            d = d.cuda(2)
            x_out = net(d)
            net.optimize_parameters(i, total_it, writer)

            writer.add_scalar('loss_G', net.loss_G.item(), total_it)
            writer.add_scalar('ssim_loss', net.ssim_loss.item(), total_it)
            writer.add_scalar('G_GAN_loss', net.loss_G_GAN.item(), total_it)
            writer.add_scalar('G_L1_loss', net.loss_G_L1.item(), total_it)
            writer.add_scalar('perc_loss', net.perc_loss.item(), total_it)
            writer.add_scalar('D_loss', net.loss_D.item(), total_it)

            if i % 10 == 0:
                print(
                    f"Train epoch, iter {epoch}, {i}: "
                    f'\tG_Loss: {net.loss_G.item():.6f} |'
                    f'\tssim_loss: {net.ssim_loss:.6f} |'
                    f"\tG_GAN_loss: {net.loss_G_GAN.item():.6f} |"
                    f"\tG_L1_loss: {net.loss_G_L1.item():.6f} |"
                    f"\tG_perc_loss: {net.perc_loss.item():.6f} |"
                    f'\tD_loss: {net.loss_D.item():.6f} |'
                )
        epoch += 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'G_optimizer_state_dict': net.optimizer_G.state_dict(),
            'D_optimizer_state_dict': net.optimizer_D.state_dict(),
            'G_loss': net.loss_G,
            'D_loss': net.loss_D,
            'total_it': total_it
        }, '/data/chenyangrui/resnetWGan/resnetWGan_checkpoints')
        torch.save(net.state_dict(), f"/data/chenyangrui/resnetWGan/resnetWGan_{epoch}.pth")


