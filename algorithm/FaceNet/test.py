import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import dataset
from endtoend import ImageCompressor
from torch import nn
from facenet_pytorch import InceptionResnetV1
from DeepRcon import DRcon

if __name__ == "__main__":
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    dataset = dataset('/data/chenyangrui/train')
    dl = DataLoader(
        dataset=dataset,
        num_workers=0,
        batch_size=2,
        shuffle=False,
        drop_last=True,
        # collate_fn=dataset.collate
    )

    # BaseLayer
    Base = DRcon().eval()
    Base = Base.cuda()
    Base = nn.DataParallel(Base).cuda()
    B_param = torch.load('/data/chenyangrui/cyr/baseLayer_6.pth')
    Base.load_state_dict(B_param)

    # EnhancementLayer
    # codec = ImageCompressor().eval()
    # codec = codec.cuda()
    # codec = nn.DataParallel(codec).cuda()
    # E_param = torch.load('../save/enhanceLayer.pth')
    # codec.load_state_dict(E_param)
    # a = 0

    with torch.no_grad():
        for data in dl:
            print(data)
            if a == 10: break
            x = data
            x = x.cuda()
            feat = resnet(x)
            feat = torch.squeeze(feat, 1)
            feat = torch.unsqueeze(feat, 2)
            feat = torch.unsqueeze(feat, 3)
            feat = feat.cuda()
            x_feat = Base(feat)

            # x_resi = x - x_feat
            # t_Max = torch.max(x_resi)
            # t_Min = torch.min(x_resi)
            # x_tex = (x_resi - t_Min) / (t_Max - t_Min)  # min-max normalization
            # x_tex.cuda()
            # x_resi.cuda()
            # decoded = codec(x_resi)
            # r_Max = torch.max(decoded)
            # r_Min = torch.min(decoded)
            # x_rec = (decoded - r_Min) / (r_Max - r_Min)
            # x_rec.cuda()
            # output = x_feat + decoded
            torchvision.utils.save_image(x, f'../result/{a}.png')
            # torchvision.utils.save_image(x_tex, f'../result/{a}_tex.png')
            # torchvision.utils.save_image(output, f'../result/{a}_output.png')
            torchvision.utils.save_image(x_feat, f'../result/{a}_feat.png')
            # torchvision.utils.save_image(x_rec, f'../result/{a}_rec.png')
            # torchvision.utils.save_image(decoded, f'../result/{a}_decoded.png')
            a = a + 1
