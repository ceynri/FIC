import torch
from torch.utils.data import DataLoader
from  dataset import dataset
from torch.utils.data.dataloader import default_collate
from network import GAN
import cv2


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
        batch_size=80,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate
    )
    net = GAN(train=True).cuda(2)
    epoch = 0

    print("start train")
    while(epoch < 30):
        net.train()
        for i, d in enumerate(ds):
            d = d.cuda(2)
            x_out = net(d)
            net.optimize_parameters()

            if i % 10 == 0:
                print(
                    f"Train epoch, iter {epoch}, {i}: "
                    f'\tG_Loss: {net.loss_G.item():.6f} |'
                    f'\tssim_loss: {net.ssim_loss:.6f} |'
                    f"\tG_GAN_loss: {net.loss_G_GAN.item():.6f} |"
                    f"\tG_L1_loss: {net.loss_G_L1.item():.6f} |"
                    f'\tD_loss: {net.loss_D.item():.6f} |'
                )
        epoch += 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'G_optimizer_state_dict': net.optimizer_G.state_dict(),
            'D_optimizer_state_dict': net.optimizer_D.state_dict(),
            'G_loss': net.loss_G,
            'D_loss': net.loss_D
        }, '/data/chenyangrui/save/BaseLayer_perc_checkpoints')
        torch.save(net.state_dict(), f"/data/chenyangrui/save/BaseLayer_perc_{epoch}.pth")


