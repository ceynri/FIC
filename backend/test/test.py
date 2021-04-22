import json
import pickle
import sys
from os import path

import torch
import torch.nn as nn
from autocrop import Cropper
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))

import config as conf
from models.base.deeprecon import DeepRecon
from models.enhancement.gdnmodel import GdnModel
from utils import load_image_array, tensor_normalize, tensor_to_array
from utils.eval import psnr, ssim
from utils.file import File
from utils.jpeg import dichotomy_compress


def get_path(filename):
    return path.join('./result', filename)


class File:
    def __init__(self, file_path):
        self.path = file_path
        self.ext = path.splitext(file_path)[1]
        self.name = path.splitext(path.basename(file_path))[0]

    def load_tensor(self):
        cropper = Cropper(face_percent=100)
        img_cropped = cropper.crop(self.path)
        img = Image.fromarray(img_cropped)
        loader = transforms.Compose(
            [transforms.Resize(conf.IMAGE_SHAPE),
             transforms.ToTensor()])
        tensor = loader(img).unsqueeze(0)
        self.tensor = tensor
        return tensor

    def load_fic(self):
        with open(self.path, 'rb') as f:
            self.fic = pickle.load(f)
            return self.fic

    def name_suffix(self, suffix: str, ext: str = ''):
        if ext == '':
            ext = self.ext
        return f'{self.name}_{suffix}{ext}'

    @classmethod
    def save_binary(cls, data: dict, file_name: str):
        with open(file_name, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        return file_name

    @classmethod
    def load_binary(cls, file, save_path: str = './public/temp/'):
        save_path = path.join(save_path, file.filename)
        file.save(save_path)
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
            return data


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
    b_layer = DeepRecon().eval()
    b_layer = b_layer.cuda()
    b_layer = nn.DataParallel(b_layer).cuda()
    b_param = torch.load('../params/recon/30w/baseLayer_7.pth',
                         map_location='cuda:0')
    b_layer.load_state_dict(b_param)

    e_layer = GdnModel().eval().cuda()
    e_layer = CustomDataParallel(e_layer).cuda()
    c_param = torch.load('../params/e_layer/5120/enhanceLayer_7.pth', map_location='cuda:0')
    # c_param = torch.load('../params/e_layer/1024/gdnmodel_10.pth', map_location='cuda:0')
    e_layer.load_state_dict(c_param)

    file_path = sys.argv[1]
    file = File(file_path)
    file.load_tensor()

    with torch.no_grad():
        input = file.tensor
        input = input.cuda()
        feat = resnet(input)

        # process feat shape to [N, 512, 1, 1]
        feat = torch.squeeze(feat, 1)
        feat = torch.unsqueeze(feat, 2)
        feat = torch.unsqueeze(feat, 3)
        feat = feat.cuda()

        # reconstruct feature image
        recon = b_layer(feat)

        # EnhancementLayer
        resi = input - recon
        resi = resi.cuda()

        # 残差纹理图压缩
        resi_norm, intervals = tensor_normalize(resi)
        resi_norm = resi_norm.cuda()
        tex = e_layer.compress(resi_norm)
        resi_decoded_norm = e_layer.decompress(tex)

        resi_decoded = tensor_normalize(resi_decoded_norm,
                                        intervals=intervals,
                                        mode='anti')
        output = recon + resi_decoded

        # 保存压缩数据
        fic_path = get_path(f'{file.name}.fic')
        File.save_binary(
            {
                'feat': feat,
                'tex': tex,
                'intervals': intervals,
                'ext': file.ext,
            }, fic_path)

        # fic 相关参数
        fic_size = path.getsize(fic_path)
        fic_bpp = fic_size / conf.IMAGE_PIXEL_NUM

        # 单独保存特征以计算特征和纹理的大小
        feat_path = get_path(f'{file.name}_feat.fic')
        File.save_binary({
            'feat': feat,
        }, feat_path)
        # 特征相关参数
        feat_size = path.getsize(feat_path)
        feat_bpp = feat_size / conf.IMAGE_PIXEL_NUM
        # 纹理相关参数
        tex_size = fic_size - feat_size
        tex_bpp = tex_size / conf.IMAGE_PIXEL_NUM

        # 待保存图片 # TODO RENAME
        imgs = {
            'input': input,
            'feat': recon,
            'tex': resi,
            'tex_decoded': resi_decoded,
            'tex_norm': resi_norm,
            'tex_decoded_norm': resi_decoded_norm,
            'output': output,
        }

        # 将 imgs 保存
        for key, value in imgs.items():
            # 保存图片
            file_name = file.name_suffix(key, ext='.bmp')
            file_path = get_path(file_name)
            save_image(value, file_path)

        # 计算压缩率
        input_name = file.name_suffix('input', ext='.bmp')
        input_path = get_path(input_name)
        input_size = path.getsize(input_path)
        fic_compression_ratio = fic_size / input_size

        # jpeg对照组处理
        jpeg_name = file.name_suffix('jpeg', ext='.jpg')
        jpeg_path = get_path(jpeg_name)
        dichotomy_compress(input_path, jpeg_path, target_size=tex_size)

        # jpeg 相关参数计算
        jpeg_size = path.getsize(jpeg_path)
        jpeg_compression_ratio = jpeg_size / input_size
        jpeg_bpp = jpeg_size / conf.IMAGE_PIXEL_NUM

        # 其他数据
        input_arr = tensor_to_array(input)
        output_arr = tensor_to_array(output)
        jpeg_arr = load_image_array(jpeg_path)

        print(json.dumps({
            'eval': {
                # 'fic_bpp': fic_bpp,
                # 'feat_bpp': feat_bpp,
                'tex_bpp': tex_bpp,
                'jpeg_bpp': jpeg_bpp,
                # 'fic_compression_ratio': fic_compression_ratio,
                # 'jpeg_compression_ratio': jpeg_compression_ratio,
                'fic_psnr': psnr(input_arr, output_arr),
                'fic_ssim': ssim(input_arr, output_arr),
                'jpeg_psnr': psnr(input_arr, jpeg_arr),
                'jpeg_ssim': ssim(input_arr, jpeg_arr),
            },
            'size': {
                'fic': fic_size,
                # 'input': input_size,
                # 'output': fic_size,
                # 'feat': feat_size,
                'tex': tex_size,
                'jpeg': jpeg_size,
            }
        }, indent=4))
