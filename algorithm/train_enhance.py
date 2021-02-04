import torch
import numpy as np
import torch.optim as optim
import multiprocessing as mp
from torch.utils.data import DataLoader
from endtoend import ImageCompressor
from torch import nn
from facenet_pytorch import InceptionResnetV1
from DeepRcon import DRcon
from dataset import dataset


def clip_gradient(optimizer, grad_clip):  # 梯度裁剪防止梯度爆炸
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    # DataLoader requires multiprocessing ?
    mp.set_start_method('spawn')

    # FaceNet for extracting features
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    resnet = nn.DataParallel(resnet, list(range(gpu_num)))

    # reconstruct face by feature
    Base = DRcon().eval()
    Base = Base.cuda()
    Base = nn.DataParallel(Base, list(range(gpu_num)))
    param = torch.load('../save/baseLayer.pth')
    Base.load_state_dict(param)

    ds = dataset()
    dl = DataLoader(
        dataset = ds,
        num_workers = 10,
        batch_size = 256,
        shuffle = True,
        drop_last = True,
        pin_memory = True,
        # collate_fn = ds.collate
    )

    # End-to-End Compression model
    codec = ImageCompressor().cuda()
    codec = nn.DataParallel(codec, list(range(gpu_num)))
    parameters = codec.parameters()
    optimizer = optim.Adam(parameters, lr=2e-4)

    # Training
    loss = 0
    for epoch in range(10):
        batch = 0
        for dt in dl:
            # BaseLayer
            label = dt
            label = label.cuda()
            feat = resnet(label)
            feat = torch.squeeze(feat,1)
            feat = torch.unsqueeze(feat,2)
            feat = torch.unsqueeze(feat,3)
            # feat's shape is [N,512,1,1]
            feat = feat.cuda()
            data = Base(feat)

            # EnhancementLayer
            resi = label - data
            Max = torch.max(resi)
            Min = torch.min(resi)
            x_tex = (resi-Min)/(Max-Min) # min-max normalization
            x_tex = x_tex.cuda()
            decoded = codec(resi)
            x_rec = (decoded-Min)/(Max-Min) #torch.Size([N, 3, 256, 256])
            x_rec = x_rec.cuda()

            loss = torch.mean((x_rec - x_tex).pow(2))
            if loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / loss) / np.log(10))
                print('psnr:', psnr)

            print("[{}, {}]: loss={}".format(epoch, batch, loss))
            optimizer.zero_grad()
            loss.backward()
            batch = batch + 1
            clip_gradient(optimizer, 5)
            optimizer.step()

        torch.save({
            'epoch':epoch,
            'model_state_dict':codec.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss
        },'../save/enhanceLayer_checkpoints')
        torch.save(codec.state_dict(), "../save/enhanceLayer.pth")
