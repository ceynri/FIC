import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import collate, dataset
from deconv_recon import DeconvRecon
from gdn_model import GdnModel
from rate_distortion_loss import RateDistortionLoss

# config
DATASET_DIR = 'path/to/your/dataset/dir'
BASE_PARAM_PATH = 'path/to/your/base_param'
LOAD_PATH = ''
LAMBDA = 2560
SAVE_DIR = f'path/to/your/save/dir/{LAMBDA}'
CUDA_IDX = 0
CUDA_LIST = [0]


def configure_optimizers(net, lr, aux_lr):
    '''Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers'''

    parameters = set(p for n, p in net.named_parameters()
                     if not n.endswith('.quantiles'))
    aux_parameters = set(p for n, p in net.named_parameters()
                         if n.endswith('.quantiles'))

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
    '''Custom DataParallel to access the module methods.'''
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)
    ds = dataset(DATASET_DIR)
    dl = DataLoader(dataset=ds,
                    num_workers=10,
                    batch_size=100,
                    shuffle=True,
                    drop_last=True,
                    pin_memory=True,
                    collate_fn=collate)

    # FaceNet for extracting features
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda(CUDA_IDX)
    resnet = nn.DataParallel(resnet, CUDA_LIST)

    # reconstruct face by feature
    Base = DeconvRecon().cuda(CUDA_IDX)
    Base = nn.DataParallel(Base, CUDA_LIST)
    param = torch.load(BASE_PARAM_PATH, map_location='cuda:0')
    Base.load_state_dict(param)

    codec = GdnModel().cuda(CUDA_IDX)
    codec = CustomDataParallel(codec, CUDA_LIST)
    optimizer, aux_optimizer = configure_optimizers(codec,
                                                    lr=1e-3,
                                                    aux_lr=1e-3)
    criterion = RateDistortionLoss(lmbda=LAMBDA)
    clip_max_norm = 1.0
    out_criterion = {}
    epoch = 0

    writer = SummaryWriter(comment=f'_gdn_model_{LAMBDA}')

    if LOAD_PATH:
        checkpoint = torch.load(LOAD_PATH)
        codec.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])
        epoch = checkpoint['epoch']

    print('start train')
    global_step = 0
    while (epoch < 10):
        codec.train()
        for i, d in enumerate(dl):
            # BaseLayer
            label = d
            label = label.cuda(CUDA_IDX)
            feat = resnet(label)
            feat = torch.squeeze(feat, 1)
            feat = torch.unsqueeze(feat, 2)
            feat = torch.unsqueeze(feat, 3)
            # feat's shape is [N,512,1,1]
            feat = feat.cuda(CUDA_IDX)
            data = Base(feat)

            aux_optimizer.zero_grad()
            optimizer.zero_grad()

            # EnhancementLayer
            resi = label - data
            Max = torch.max(resi)
            Min = torch.min(resi)
            tex_norm = (resi - Min) / (Max - Min)  # min-max normalization
            tex_norm = tex_norm.cuda(CUDA_IDX)
            decoded = codec(tex_norm)
            x_hat = decoded['x_hat']
            recon = (Max - Min) * x_hat + Min

            out_criterion = criterion(decoded, resi, tex_norm, x_hat)
            out_criterion['loss'].backward()

            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(codec.parameters(),
                                               clip_max_norm)
            optimizer.step()

            aux_loss = codec.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            # 记录loss
            loss = out_criterion["loss"].item()
            mse_loss = out_criterion["mse_loss"].item()
            bpp_loss = out_criterion["bpp_loss"].item()
            aux_loss = aux_loss.item()
            writer.add_scalar('loss/total', loss, global_step)
            writer.add_scalar('loss/mse', mse_loss, global_step)
            writer.add_scalar('loss/bpp', bpp_loss, global_step)
            writer.add_scalar('loss/aux', aux_loss, global_step)
            global_step += 1

            if i % 10 == 0:
                print(f'Epoch {epoch}, batch {i}\t| '
                      f'Loss: {loss:.3f}\t| '
                      f'MSE loss: {mse_loss:.5f}\t| '
                      f'Bpp loss: {bpp_loss:.4f}\t| '
                      f'Aux loss: {aux_loss:.2f}')

        epoch += 1
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': codec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'aux_optimizer_state_dict': aux_optimizer.state_dict(),
                'loss': out_criterion
            }, f'{SAVE_DIR}/gdn_model_checkpoints')
        torch.save(codec.state_dict(), f'{SAVE_DIR}/gdn_model_{epoch}.pth')
