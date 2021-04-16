
from io import BytesIO
from os import path

import numpy as np
from autocrop import Cropper
from PIL import Image
from torchvision import transforms
import pickle


class File:
    def __init__(self, file):
        self.raw_file = file
        self.bytes = file.read()
        self.name, self.ext = path.splitext(file.filename)

    def load_tensor(self):
        stream = BytesIO(self.bytes)
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
