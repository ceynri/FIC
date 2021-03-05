from io import BytesIO
from PIL import Image
from autocrop import Cropper
import torch
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
        split_name = path.splitext(file.filename)
        self.name = split_name[0]
        self.ext = split_name[1]
        # self.name, self.ext = split_name

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

    def name_suffix(self, suffix):
        return f'{self.name}_{suffix}{self.ext}'


def save_compressed_data(feat, tex, file_name):
    data = {
        'feat': feat,
        'tex': tex,
    }
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def tensor_normalize(tensor):
    max = torch.max(tensor)
    min = torch.min(tensor)
    return (tensor - min) / (max - min)
