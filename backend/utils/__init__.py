import config as conf
import numpy as np
import torch
from PIL import Image
from torch import nn


def load_image_array(path: str):
    # 使用PIL读取
    img = Image.open(path)  # PIL.Image.Image对象
    return np.array(img, dtype=np.int16)


def tensor_to_array(tensor: torch.Tensor):
    img = tensor.mul(255)
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0)).astype(np.int16)
    return img


def tensor_normalize(
    tensor: torch.Tensor, intervals=None, mode: str = "normal"
):
    min, max = 0, 0
    if intervals is None:
        min, max = torch.min(tensor), torch.max(tensor)
    else:
        min, max = intervals
    if mode != 'anti':
        return ((tensor - min) / (max - min)), (min, max)
    return tensor * (max - min) + min


def get_bpp(size):
    return size * 8 / conf.IMAGE_PIXEL_NUM


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
