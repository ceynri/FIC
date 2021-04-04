import pickle
import sys
from os import path

import torch
import torch.nn as nn
from autocrop import Cropper
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from gan.network import GAN

base_path = './test'


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
    net = GAN(train=True).cuda()
    param = torch.load('./data/BaseLayer.pth', map_location='cuda:0')
    net.load_state_dict(param)

    file_path = sys.argv[1]
    file = File(file_path)
    file.load_tensor()

    with torch.no_grad():
        input = file.tensor
        input = input.cuda()
        # feat = encoder(input)
        # feat = feat.cuda()

        # # reconstruct feature image
        # output = decoder(feat)
        output = net(input)
        save_image(input, path.join(base_path, file.name_suffix('_input')))
        save_image(output, path.join(base_path, file.name_suffix('_output')))
