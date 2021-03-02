from os import path
import sys
import numpy as np

import torch
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image

from autocrop import Cropper

# from torch.utils.data import DataLoader
# from dataset import dataset

from PIL import Image

from facenet_pytorch import InceptionResnetV1

from base_layer import BaseLayer
from enhancement_layer import ImageCompressor


base_path = '../public/result'


def load_tensor(img_path):
    # img = Image.open(img_path).convert('RGB')
    cropper = Cropper()
    img_cropped = cropper.crop(img_path)
    img = Image.fromarray(img_cropped)
    loader = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    tensor = loader(img).unsqueeze(0)
    return tensor


def save_compressed_data(tensor, name):
    file_name = path.join(base_path, name)
    data = np.array(tensor.numpy())  # tensor转换成array
    np.savetxt(file_name, data)


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
    img_tensor = load_tensor(img_path)

    with torch.no_grad():
        # 输入图像
        x = img_tensor
        x = x.cuda()
        # 获取特征
        feat = resnet(x)  # 1x512
        feat = torch.squeeze(feat, 1)  # 1x512
        feat = torch.unsqueeze(feat, 2)  # 1x512x1
        feat = torch.unsqueeze(feat, 3)  # 1x512x1x1
        feat = feat.cuda()

        # 特征重建图像
        x_feat = b_layer(feat)
        # 残差纹理图
        x_resi = x - x_feat
        x_resi = x_resi.cuda()
        # 残差图归一化
        t_Max = torch.max(x_resi)
        t_Min = torch.min(x_resi)
        x_resi_norm = (x_resi - t_Min) / (t_Max - t_Min)
        # x_resi_norm.cuda()
        # 纹理图压缩+解压
        x_recon = e_layer(x_resi)
        # 纹理图归一化
        r_Max = torch.max(x_recon)
        r_Min = torch.min(x_recon)
        x_recon_norm = (x_recon - r_Min) / (r_Max - r_Min)
        # x_recon_norm.cuda()
        # 结果图
        output = x_feat + x_recon

        result = [
            [x, f'{img_name}_input{img_ext}'],
            [x_feat, f'{img_name}_feat{img_ext}'],
            [x_resi, f'{img_name}_resi{img_ext}'],
            [x_resi_norm, f'{img_name}_resi_norm{img_ext}'],
            [x_recon, f'{img_name}_recon{img_ext}'],
            [x_recon_norm, f'{img_name}_recon_norm{img_ext}'],
            [output, f'{img_name}_output{img_ext}'],
        ]
        for item in result:
            save_image(item[0], path.join(base_path, item[1]))
