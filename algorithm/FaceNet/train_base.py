import torch
import torch.optim as optim
import multiprocessing as mp
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from dataset import dataset
from DeepRcon import DRcon
from perceptual_loss import Perc
from torch.optim.lr_scheduler import ExponentialLR


def cosine_score(label, pred):
    return 1.0 - ((label - pred) ** 2.0).mean() * label.shape[1] / 2.


if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    # multi core process
    mp.set_start_method('spawn')

    # FaceNet for extracting features
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    resnet = torch.nn.DataParallel(resnet, list(range(gpu_num)))

    dataset = dataset()
    dl = DataLoader(
        dataset=dataset,
        num_workers=10,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        # collate_fn=dataset.collate
    )

    model = DRcon()
    model = model.cuda()
    model = torch.nn.DataParallel(model, list(range(gpu_num)))

    criterion = torch.nn.L1Loss()
    # perceptual loss
    perc = Perc()
    perc = perc.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ExponentialLR(optimizer, gamma=0.9125)

    loss = 0
    for epoch in range(0, 50):
        batch = 0
        for data in dl:
            # extract 512 dim featues
            label = data
            label = label.cuda()
            feat = resnet(label)

            feat = torch.squeeze(feat, 1)
            feat = torch.unsqueeze(feat, 2)
            feat = torch.unsqueeze(feat, 3)
            # feat's shape is [N,512,1,1]
            feat = feat.cuda()

            # label's shape is [N,3,256,256]
            recon = model(feat)

            loss1 = criterion(recon, label)
            loss2 = perc(recon, label)
            loss = loss1 + 0.00001 * loss2

            # [epoch, batch]: loss, score
            print("[{}, {}]: loss={}, score={}, ".format(epoch, batch, loss, cosine_score(label, recon)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch = batch + 1

        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, '../save/baseLayer_checkpoints')
        torch.save(model.state_dict(), "../save/baseLayer.pth")
    print("train finished!")
