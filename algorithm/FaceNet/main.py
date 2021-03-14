import torch
import torch.nn as nn
import torch.optim as optim
from dataset import dataset, collate
from torch.utils.data import DataLoader
from endtoend import AutoEncoder
from RateDistortionLoss import RateDistortionLoss
from facenet_pytorch import InceptionResnetV1
from DeepRcon import DRcon
import cv2

def configure_optimizers(net, lr, aux_lr):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        p for n, p in net.named_parameters() if not n.endswith(".quantiles")
    )
    aux_parameters = set(
        p for n, p in net.named_parameters() if n.endswith(".quantiles")
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (p for p in parameters if p.requires_grad),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (p for p in aux_parameters if p.requires_grad),
        lr=aux_lr,
    )
    return optimizer, aux_optimizer

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

if __name__ == '__main__':
    # isload = True
    # gpu_num = torch.cuda.device_count()
    cv2.ocl.setUseOpenCL(False)
    ds = dataset('/data/chenyangrui/train')
    dl = DataLoader(
        dataset=ds,
        num_workers=10,
        batch_size=100,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn = collate
    )

    # FaceNet for extracting features
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    #resnet = nn.DataParallel(resnet, list(range(gpu_num)))
    resnet = nn.DataParallel(resnet, [0,2])

    # reconstruct face by feature
    Base = DRcon().eval()
    Base = Base.cuda()
    Base = nn.DataParallel(Base, [0,2])
    param = torch.load('/data/chenyangrui/save/base_layer.pth')
    Base.load_state_dict(param)

    codec = AutoEncoder().cuda()
    codec = CustomDataParallel(codec, [0,2])
    optimizer, aux_optimizer = configure_optimizers(codec, lr=1e-3, aux_lr=1e-3)
    criterion = RateDistortionLoss(lmbda=1024)
    clip_max_norm = 1.0
    out_criterion = {}
    epoch = 0

    # if isload:
    #     checkpoint = torch.load("/data/chenyangrui/save/enhanceLayer_checkpoints")
    #     codec.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     aux_optimizer.load_state_dict(checkpoint["aux_optimizer_state_dict"])
    #     epoch = checkpoint["epoch"]


    print("start train")
    while(epoch < 10):
        codec.train()
        for i, d in enumerate(dl):
            # BaseLayer
            label = d
            label = label.cuda()
            feat = resnet(label)
            feat = torch.squeeze(feat, 1)
            feat = torch.unsqueeze(feat, 2)
            feat = torch.unsqueeze(feat, 3)
            # feat's shape is [N,512,1,1]
            feat = feat.cuda()
            data = Base(feat)

            aux_optimizer.zero_grad()
            optimizer.zero_grad()

            # EnhancementLayer
            resi = label - data
            Max = torch.max(resi)
            Min = torch.min(resi)
            tex_norm = (resi - Min) / (Max - Min)  # min-max normalization
            tex_norm = tex_norm.cuda()
            decoded = codec(tex_norm)
            x_hat = decoded["x_hat"]
            recon = (Max - Min) * x_hat + Min

            out_criterion = criterion(decoded, resi, tex_norm, x_hat)
            out_criterion["loss"].backward()

            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(), clip_max_norm)
            optimizer.step()

            aux_loss = codec.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            if i%10==0:
                print(
                    f"Train epoch, iter {epoch}, {i}: "
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )
        epoch += 1
        torch.save({
            'epoch': epoch,
            'model_state_dict': codec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'aux_optimizer_state_dict': aux_optimizer.state_dict(),
            'loss': out_criterion
        }, '/data/chenyangrui/save/enhanceLayer_checkpoints')
        torch.save(codec.state_dict(), "/data/chenyangrui/save/enhanceLayer_lamda.pth")
