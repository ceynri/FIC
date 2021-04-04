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

sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))

from fcn.endtoend import AutoEncoder
from fcn.DeepRcon import DRcon
from utils import save_binary_file, tensor_normalize

base_path = './test/result'


def get_path(filename):
    return path.join(base_path, filename)


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
    b_layer = DRcon().eval()
    b_layer = b_layer.cuda()
    b_layer = nn.DataParallel(b_layer).cuda()
    b_param = torch.load('./data/b_layer_fcn.pth', map_location='cuda:0')
    b_layer.load_state_dict(b_param)

    e_layer = AutoEncoder().eval().cuda()
    e_layer = CustomDataParallel(e_layer).cuda()
    c_param = torch.load('./data/e_layer_5120.pth', map_location='cuda:0')
    e_layer.load_state_dict(c_param)

    file_path = sys.argv[1]
    file = File(file_path)
    file.load_tensor()

    with torch.no_grad():
        x_input = file.tensor
        x_input = x_input.cuda()
        feat = resnet(x_input)

        # process feat shape to [N, 512, 1, 1]
        feat = torch.squeeze(feat, 1)
        feat = torch.unsqueeze(feat, 2)
        feat = torch.unsqueeze(feat, 3)
        feat = feat.cuda()

        # reconstruct feature image
        x_feat = b_layer(feat)

        # EnhancementLayer
        x_resi = x_input - x_feat
        x_resi = x_resi.cuda()

        # 残差纹理图压缩
        x_resi_norm, intervals = tensor_normalize(x_resi)
        x_resi_norm = x_resi_norm.cuda()
        tex = e_layer.compress(x_resi_norm)

        x_resi_norm = tensor_normalize(x_resi)

        # 保存压缩数据
        fic_path = get_path(f'{file.name}.fic')
        save_binary_file({
            'feat': feat,
            'tex': tex,
            'intervals': intervals,
            'ext': file.ext,
        }, fic_path)
