from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from PIL import Image
from pathlib import Path
from autocrop import Cropper
from rich.progress import track

def collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    while len(batch) < 16:
        j = len(batch)
        for i in (0, j):
            if len(batch) < 16:
                batch.append(batch[i])
    return default_collate(batch)

class dataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.image_files = []
        self.crop = Cropper(face_percent=100)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        print("begin loading")
        for pdir in track(list(self.path.glob('*/*.jpg'))):
            # use path index image
            self.image_files.append(str(pdir))
        # self.image_files = self.image_files[:300000]
        #print('finish loading')

    def __len__(self):
        return (len(self.image_files))

    def __getitem__(self, index):
        cropped = self.crop.crop(self.image_files[index])
        if cropped is None:
            return None
        img = Image.fromarray(cropped).convert('RGB')
        img = self.transform(img)
        return img
