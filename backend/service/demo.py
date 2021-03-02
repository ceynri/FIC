from os import path
import sys

import torch
from torch import nn

from torchvision import transforms
from torchvision.utils import save_image

# from torch.utils.data import DataLoader
# from dataset import dataset

from PIL import Image

from facenet_pytorch import InceptionResnetV1

from base_layer import BaseLayer
from enhancement_layer import ImageCompressor


def load_regularize_data(img_path):
    img = Image.open(img_path).convert('RGB')
    loader = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = loader(img).unsqueeze(0)
    return image


if __name__ == "__main__":
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

    # BaseLayer
    b_layer = BaseLayer().eval()
    b_layer = b_layer.cuda()
    b_layer = nn.DataParallel(b_layer).cuda()
    b_param = torch.load('./data/base_layer.pth')
    b_layer.load_state_dict(b_param)

    # EnhancementLayer
    e_layer = ImageCompressor().eval()
    e_layer = e_layer.cuda()
    e_layer = nn.DataParallel(e_layer).cuda()
    e_param = torch.load('./data/enhancement_layer.pth')
    e_layer.load_state_dict(e_param)

    img_path = sys.argv[1]
    img_ext = path.splitext(img_path)[1]
    img_name = path.splitext(path.basename(img_path))[0]
    data = load_regularize_data(img_path)

    with torch.no_grad():
        x = data
        x = x.cuda()
        feat = resnet(x)
        feat = torch.squeeze(feat, 1)
        feat = torch.unsqueeze(feat, 2)
        feat = torch.unsqueeze(feat, 3)
        feat = feat.cuda()
        x_feat = b_layer(feat)

        x_resi = x - x_feat
        t_Max = torch.max(x_resi)
        t_Min = torch.min(x_resi)
        # min-max normalization
        x_tex = (x_resi - t_Min) / (t_Max - t_Min)
        x_tex.cuda()
        x_resi.cuda()
        decoded = e_layer(x_resi)
        r_Max = torch.max(decoded)
        r_Min = torch.min(decoded)
        x_rec = (decoded - r_Min) / (r_Max - r_Min)
        x_rec.cuda()
        output = x_feat + decoded

        base_path = '../public/result'
        result = [
            [x, f'{base_path}/{img_name}{img_ext}'],
            [x_tex, f'{base_path}/{img_name}_tex{img_ext}'],
            [output, f'{base_path}/{img_name}_output{img_ext}'],
            [x_feat, f'{base_path}/{img_name}_feat{img_ext}'],
            [x_rec, f'{base_path}/{img_name}_rec{img_ext}'],
            [decoded, f'{base_path}/{img_name}_decoded{img_ext}'],
        ]
        for item in result:
            save_image(item[0], item[1])

        print(result[:1])
