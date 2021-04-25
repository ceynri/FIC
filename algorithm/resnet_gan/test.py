import torch
from network import GAN
import cv2
from torch.utils.data import DataLoader
from  dataset import testDataset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from autocrop import Cropper
from rich.progress import track
import numpy as np
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

def collate(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate(batch)

if __name__ == '__main__':
    cv2.ocl.setUseOpenCL(False)
    class_num = 15
    dataset = testDataset('/data/chenyangrui/train', class_num=class_num)
    ds = DataLoader(
        dataset=dataset,
        num_workers=10,
        batch_size=40,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate
    )

    standard = []
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    Crop = Cropper(face_percent=100)
    path = Path('/data/chenyangrui/train')
    for name in list(path.glob('*'))[:class_num]:
        images = list(name.glob('*.jpg'))
        for img in images:
            cropped = Crop.crop(str(img))
            if cropped is not None:
                img = Image.fromarray(cropped).convert('RGB')
                img = transform(img)
                standard.append(img)
                break


    matrix = np.zeros((class_num, class_num), dtype=np.int32)
    s_features = []
    # net = GAN(train=False).cuda(3)
    # param = torch.load("/data/chenyangrui/resnetGan/resnetGan.pth")
    # net.netG.load_state_dict(param)
    net = InceptionResnetV1(pretrained='vggface2').cuda(3).eval()

    criterion = nn.MSELoss()

    for each in standard:
        net.eval()
        x = torch.unsqueeze(each, 0).cuda(3)
        # feature, _ = net(x)
        feature = net(x)
        s_features.append(feature)

    print('test start')
    for i, d in enumerate(ds):
        net.eval()
        (imgs, labels) = d
        imgs = imgs.cuda(3)
        # features, _ = net(imgs)
        features = net(imgs)
        for i in range(len(features)):
            loss = []
            for f in s_features:
                loss.append(criterion(f[0], features[i]))
            idx = torch.argmin(torch.Tensor(loss))
            matrix[labels[i]][idx] += 1
        print(matrix)
    print(np.sum(matrix))
        