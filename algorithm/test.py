import torch
import os
import torchvision
import torch.optim as optim
import PIL.Image as pil_image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from endtoend import ImageCompressor
from glob import glob
from torch import nn
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR
from facenet_pytorch import InceptionResnetV1
from reconstruction1 import Dconv
# from bit_estimate import BitEstimator
from autocrop import Cropper
from torch.utils.data.dataloader import default_collate


def collate(batch): 
    batch = list(filter(lambda x:x is not None, batch))
    while len(batch) < 2:
        j = len(batch)
        for i in (0,j):
            if len(batch) < 2:
                batch.append(batch[i])
    return default_collate(batch)


class ds(Dataset):
    def __init__(self,cropper):
        path = '../data/train-data/train'
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

if __name__ == "__main__":
    cropper = Cropper()
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    dataset = ds(cropper)
    dl = DataLoader(
        dataset = dataset,
        num_workers = 0,
        batch_size = 2,
        shuffle = False,
        drop_last = True,
        collate_fn = collate
    )

    # BaseLayer
    Base = Dconv().eval()
    Base = Base.cuda()
    Base = nn.DataParallel(Base).cuda()
    D_param = torch.load('../recon_epoch_30.pth')
    Base.load_state_dict(D_param)

    # EnhancementLayer
    codec = ImageCompressor().eval()
    codec = codec.cuda()
    codec = nn.DataParallel(codec).cuda()
    E_param = torch.load('../enhancement_epoch_9.pth')
    codec.load_state_dict(E_param)
    a = 0

    with torch.no_grad():
        for data in dl:
            print(data)
            if a == 10: break
            x = data
            x = x.cuda()            
            feat = resnet(x)            
            feat = torch.squeeze(feat,1)
            feat = torch.unsqueeze(feat,2)
            feat = torch.unsqueeze(feat,3)
            feat = feat.cuda()  
            x_feat = Base(feat)

            x_resi = x - x_feat
            t_Max = torch.max(x_resi)
            t_Min = torch.min(x_resi)
            x_tex = (x_resi - t_Min) / (t_Max - t_Min) # min-max normalization
            x_tex.cuda()
            x_resi.cuda()
            decoded = codec(x_resi)
            r_Max = torch.max(decoded)
            r_Min = torch.min(decoded)
            x_rec = (decoded - r_Min) / (r_Max - r_Min)
            x_rec.cuda()
            output = x_feat + decoded
            torchvision.utils.save_image(x, '../result/' + str(a) + '.png')
            torchvision.utils.save_image(x_tex, '../result/' + str(a) + '_tex.png')
            torchvision.utils.save_image(output, '../result/' + str(a) + '_output.png')
            torchvision.utils.save_image(x_feat, '../result/' + str(a) + '_feat.png')
            torchvision.utils.save_image(x_rec, '../result/' + str(a) + '_rec.png')
            torchvision.utils.save_image(decoded, '../result/' + str(a) + '_decoded.png')
            a = a + 1

