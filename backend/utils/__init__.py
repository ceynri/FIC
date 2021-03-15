from io import BytesIO
from PIL import Image
from autocrop import Cropper
import torch
from torch import nn
from torchvision import transforms
import numpy as np
import pickle
from os import path

# def bytes_tensor_loader(btye):
#     img_array = np.frombuffer(btye, dtype=np.uint8)
#     cropper = Cropper()
#     img_cropped = cropper.crop(img_array)
#     img = Image.fromarray(img_cropped)
#     loader = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])
#     tensor = loader(img).unsqueeze(0)
#     return tensor


class File:
    def __init__(self, file):
        self.raw_file = file
        self.btyes = file.read()
        self.name, self.ext = path.splitext(file.filename)

    def load_tensor(self):
        stream = BytesIO(self.btyes)
        img = Image.open(stream)
        img_array = np.array(img)
        # RGB to BGR for cropper
        img_array = img_array[:, :, [2, 1, 0]]

        cropper = Cropper()
        img_cropped = cropper.crop(img_array)
        if img_cropped is None:
            raise 'No face recognized!'
        img = Image.fromarray(img_cropped)

        loader = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        tensor = loader(img).unsqueeze(0)
        self.tensor = tensor
        return tensor

    def name_suffix(self, suffix: str, ext: str = ''):
        if ext == '':
            ext = self.ext
        return f'{self.name}_{suffix}{ext}'


def save_fic(feat, tex, file_name):
    data = {
        'feat': feat,
        'tex': tex,
    }
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    return file_name


def load_image_array(path):
    # 使用PIL读取
    img = Image.open(path)  # PIL.Image.Image对象
    return np.array(img, dtype=np.int16)


def tensor_to_array(tensor):
    img = tensor.mul(255)
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.int16)
    return img


def tensor_normalize(tensor, intervals=None, mode="normal"):
    min, max = 0, 0
    if intervals is None:
        min, max = torch.min(tensor), torch.max(tensor)
    else:
        min, max = intervals
    if mode != 'anti':
        return ((tensor - min) / (max - min)), (min, max)
    return tensor * (max - min) + min


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
