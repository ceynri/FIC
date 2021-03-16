import sys
from os import path
import pickle
from PIL import Image

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from autocrop import Cropper
from torchvision import transforms
from torchvision.utils import save_image

from endtoend import AutoEncoder
from DeepRcon import DRcon
from utils import save_binary_file, tensor_normalize

base_path = './public/result'


class File:
    def __init__(self, file_path):
        self.path = file_path
        self.ext = path.splitext(file_path)[1]
        self.name = path.splitext(path.basename(file_path))[0]

    def load_tensor(self):
        cropper = Cropper()
        img_cropped = cropper.crop(self.path)
        img = Image.fromarray(img_cropped)
        loader = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        tensor = loader(img).unsqueeze(0)
        self.tensor = tensor
        return tensor

    def load_fic(self):
        with open(self.path, 'rb') as f:
            self.fic = pickle.load(f)
            return self.fic

    def name_suffix(self, suffix):
        return f'{self.name}{suffix}{self.ext}'


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


if __name__ == '__main__':
    # FaceNet for extracting features
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

    # reconstruct face by feature
    base = DRcon().eval()
    base = base.cuda()
    base = nn.DataParallel(base).cuda()
    param = torch.load('./data/base_layer.pth')
    base.load_state_dict(param)

    codec = AutoEncoder().eval().cuda()
    codec = CustomDataParallel(codec).cuda()
    c_param = torch.load('./data/enhanceLayer_lamda.pth')
    codec.load_state_dict(c_param)

    file_path = sys.argv[1]
    file = File(file_path)
    file.load_tensor()

    with torch.no_grad():
        label = file.tensor
        label = label.cuda()
        feat = resnet(label)

        # process feat shape to [N, 512, 1, 1]
        feat = torch.squeeze(feat, 1)
        feat = torch.unsqueeze(feat, 2)
        feat = torch.unsqueeze(feat, 3)
        feat = feat.cuda()

        # reconstruct feature image
        x_feat = base(feat)

        # EnhancementLayer
        x_resi = label - x_feat
        x_resi = x_resi.cuda()
        # x_recon = codec(x_resi)
        tex = codec.compress(x_resi)
        x_recon = codec.decompress(tex)
        x_output = x_feat + x_recon

        x_resi_norm = tensor_normalize(x_resi)
        x_recon_norm = tensor_normalize(x_recon)

        save_path_name = path.join(base_path, f'{file.name}.fic')
        save_binary_file(feat, tex, save_path_name)

        result = [[label, file.name_suffix('_input')],
                  [x_feat, file.name_suffix('_feat')],
                  [x_resi, file.name_suffix('_resi')],
                  [x_recon, file.name_suffix('_recon')],
                  [x_resi_norm, file.name_suffix('_resi_norm')],
                  [x_recon_norm, file.name_suffix('_recon_norm')],
                  [x_output, file.name_suffix('_output')]]
        for item in result:
            save_image(item[0], path.join(base_path, item[1]))

        data_file = File(path.join(base_path, f'{file.name}.fic'))
        data_file.load_fic()
        feat = data_file.fic['feat'].cuda()
        tex = data_file.fic['tex']
        x_feat = base(feat)
        x_recon = codec.decompress(tex)
        save_image(
            x_feat + x_recon,
            path.join(base_path, file.name_suffix('_decompressed'))
        )
