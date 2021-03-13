import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import dataset
from endtoend import ImageCompressor
from torch import nn
from facenet_pytorch import InceptionResnetV1
from DeepRcon import DRcon
# from bit_estimate import BitEstimator

def reload():
    #转旧版本pth
    pth = './enhancement_epoch_9.pth'
    checkpoint = torch.load(pth)
    codec = ImageCompressor().eval()
    codec = codec.cuda()
    codec = nn.DataParallel(codec).cuda()
    codec.load_state_dict(checkpoint)
    torch.save(codec.state_dict(), "./enhanceLayer.pth", _use_new_zipfile_serialization=False)

if __name__ == "__main__":
    #reload()
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    dataset = dataset()
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
    B_param = torch.load('../save/baseLayer.pth')
    Base.load_state_dict(B_param)

    # EnhancementLayer
    codec = ImageCompressor().eval()
    codec = codec.cuda()
    codec = nn.DataParallel(codec).cuda()
    E_param = torch.load('../save/enhanceLayer.pth')
    codec.load_state_dict(E_param)
    a = 0

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

            x_resi = x - x_feat
            t_Max = torch.max(x_resi)
            t_Min = torch.min(x_resi)
            x_tex = (x_resi - t_Min) / (t_Max - t_Min)  # min-max normalization
            x_tex.cuda()
            x_resi.cuda()
            decoded = codec(x_resi)
            r_Max = torch.max(decoded)
            r_Min = torch.min(decoded)
            x_rec = (decoded - r_Min) / (r_Max - r_Min)
            x_rec.cuda()
            output = x_feat + decoded
            torchvision.utils.save_image(x, '../result/' + str(a) + '.png')
            torchvision.utils.save_image(x_tex, '../result/' + str(a) + '_tex.png')
            torchvision.utils.save_image(output, '../result/' + str(a) + '_output.png')
            torchvision.utils.save_image(x_feat, '../result/' + str(a) + '_feat.png')
            torchvision.utils.save_image(x_rec, '../result/' + str(a) + '_rec.png')
            torchvision.utils.save_image(decoded, '../result/' + str(a) + '_decoded.png')
            a = a + 1
