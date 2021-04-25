import torch
import torch.optim as optim
import multiprocessing as mp
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from dataset import dataset, collate
from DeepRcon import DRcon
from perceptual_loss import Perc
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image


def cosine_score(label, pred):
    return 1.0 - ((label - pred) ** 2.0).mean() * label.shape[1] / 2.


if __name__ == '__main__':
    cuda_idx = 6
    # multi core process
    mp.set_start_method('spawn')

    # FaceNet for extracting features
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda(cuda_idx)
    resnet = torch.nn.DataParallel(resnet, [cuda_idx])

    ds = dataset('/data/chenyangrui/train')
    dl = DataLoader(
        dataset=ds,
        num_workers=10,
        batch_size=40,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate
    )


    model = DRcon()
    model = model.cuda(cuda_idx)

    model = torch.nn.DataParallel(model, [cuda_idx])

    criterion = torch.nn.L1Loss()
    perc = Perc()
    perc = perc.cuda(cuda_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.9125)

    writer = SummaryWriter(comment='')

    loss = 0
    global_step = 0
    for epoch in range(1, 30):
        batch = 0
        for data in dl:
            # extract 512 dim featues
            label = data
            label = label.cuda(cuda_idx)
            feat = resnet(label)

            feat = torch.squeeze(feat, 1)
            feat = torch.unsqueeze(feat, 2)
            feat = torch.unsqueeze(feat, 3)
            # feat's shape is [N,512,1,1]
            feat = feat.cuda(cuda_idx)

            # label's shape is [N,3,256,256]
            recon = model(feat)

            loss1 = criterion(recon, label)
            loss2 = perc(recon, label)
            loss = loss1 + 0.00001 * loss2

            score = cosine_score(label, recon)
            # [epoch, batch]: loss, score
            print(f'[{epoch}, {batch}]: loss={loss}, score={score}')
            writer.add_scalar('loss', loss, global_step)
            writer.add_scalar('score', score, global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch += 1
            global_step += 1


        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, '/data/chenyangrui/cyr/baseLayer_checkpoints')
        torch.save(model.state_dict(), f'/data/chenyangrui/cyr/30w/baseLayer_{epoch}.pth')
    print('train finished!')
