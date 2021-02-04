from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from PIL import Image
from pathlib import Path
from autocrop import Cropper
from rich.progress import track
from p_tqdm import p_umap


class dataset(Dataset):
    def __init__(self):
        self.path = Path('../data/train')
        self.save = Path('../data/process/')
        self.image_files = []
        self.crop = Cropper()
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        # crop face image
        if not self.save.exists():
            print("begin cropping")
            self.save.mkdir()
            self.process()
            print("finish cropping")

        # load face image
        print("begin loading")
        for pdir in track(list(self.save.glob('*/*.jpg'))):
            # load data into memory
            # img = Image.open(pdir)
            # self.image_files.append(img.copy())

            # use path index image
            self.image_files.append(pdir)
        print('finish loading')

    def process(self):
        dir = list(self.path.iterdir())
        mp = p_umap(self.cropFace, dir, num_cpus=4)

    def cropFace(self, dir):
        for pdir in dir.glob('*.jpg'):
            cropped = self.crop.crop(str(pdir))
            if cropped is not None:
                cropped_image = Image.fromarray(cropped)
                filename = Path(str(pdir).replace('train', 'process'))
                savedir = filename.parent
                if not savedir.exists():
                    savedir.mkdir()
                cropped_image.save(filename)

    # fullfill batch if face detection fails
    def collate(batch, size):
        batch = list(filter(lambda x: x is not None, batch))
        while len(batch) < size:
            j = len(batch)
            for i in (0, j):
                if len(batch) < size:
                    batch.append(batch[i])
        return default_collate(batch)

    def __getitem__(self, index):
        cropped_image = self.transform(self.image_files[index])
        return cropped_image

    def __len__(self):
        return (len(self.image_files))


if __name__ == '__main__':
    dataset()
