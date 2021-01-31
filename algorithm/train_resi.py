import torch
import os
import numpy as np
import torch.optim as optim
import PIL.Image as pil_image
import multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from endtoend import ImageCompressor
from glob import glob
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
from facenet_pytorch import InceptionResnetV1
from autocrop import Cropper
from reconstruction1 import Dconv
from torch.utils.data.dataloader import default_collate

def collate(batch): 
    batch = list(filter(lambda x:x is not None, batch))
    while len(batch) < 16:
        j = len(batch)
        for i in (0,j):
            if len(batch) < 16:
                batch.append(batch[i])
    return default_collate(batch)

class Ds(Dataset):
    def __init__(self,cropper):   
        path = '/data/chenminghui/dataset/vggface2/train'
        dirs = os.listdir(path)
        sorted(dirs)
        self.image_files = []
        for f in dirs:
            pic_dir = os.path.join(path,f)
            for img in os.listdir(pic_dir):
                pdir = os.path.join(pic_dir,img)
                self.image_files.append(pdir)
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
        self.crop = cropper

    def __getitem__(self, index):
        cropped = self.crop.crop(self.image_files[index])
        if cropped is None:
            return None
        cropped_image = Image.fromarray(cropped).convert('RGB')
        cropped_image = self.transform(cropped_image)
        return cropped_image

    def __len__(self):
        return(len(self.image_files))

if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    mp.set_start_method('spawn')

    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    resnet = nn.DataParallel(resnet, list(range(gpu_num)))
    cropper = Cropper()
    Base = Dconv().eval()
    Base = Base.cuda()
    Base = nn.DataParallel(Base, list(range(gpu_num)))
    param = torch.load('/data/chenminghui/recon/data/recon_epoch_21.pth')
    Base.load_state_dict(param)    
    
    ds = Ds(cropper)
    dl = DataLoader(
        dataset = ds,
        num_workers = 7,
        batch_size = 128,
        shuffle = True,
        drop_last = True,
        pin_memory = True,
        collate_fn = collate
    )

    # End-to-End Compression model
    codec = ImageCompressor().cuda()
    codec = nn.DataParallel(net, list(range(gpu_num)))
    parameters = codec.parameters()
    optimizer = optim.Adam(parameters, lr=2e-4)

    # Training
    for epoch in range(10):
        batch = 0
        for dt in dl:
            print("epoch: ", epoch)
            print("batch: ", batch) 
            
            # BaseLayer
            label = dt
            label = label.cuda()
            feat = resnet(label)
            feat = torch.squeeze(feat,1)
            feat = torch.unsqueeze(feat,2)
            feat = torch.unsqueeze(feat,3)
            # feat's shape is [N,512,1,1]
            feat = feat.cuda()
            data = model(feat)
            
            # EnhancementLayer
            resi = label - data
            Max = torch.max(resi)
            Min = torch.min(resi)
            x_tex = (resi-Min)/(Max-Min) # min-max normalization
            x_tex = x_tex.cuda()            
            decoded = net(resi)
            x_rec = (decoded-Min)/(Max-Min) #torch.Size([N, 3, 256, 256])
            x_rec = x_rec.cuda()

            loss = torch.mean((x_rec - x_tex).pow(2))
            if loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / loss) / np.log(10))
                print('psnr:', psnr)             
            # print('bpp:', bpp)
            print('loss:', loss)
            optimizer.zero_grad()
            loss.backward()
            batch = batch + 1

            def clip_gradient(optimizer, grad_clip): # 梯度裁剪防止梯度爆炸
                for group in optimizer.param_groups:
                    for param in group["params"]:
                        if param.grad is not None:
                            param.grad.data.clamp_(-grad_clip, grad_clip)            
            clip_gradient(optimizer, 5)
            
            optimizer.step()
        torch.save({
            'epoch':epoch,
            'model_state_dict':net.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss
        },'/data/chenminghui/recon/end/data/checkpoints')
        torch.save(net.state_dict(), os.path.join("/data/chenminghui/recon/end/data", '{}_epoch_{}.pth'.format("enhancement", epoch)))
