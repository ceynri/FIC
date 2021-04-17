import config as conf
import torch
from facenet_pytorch import InceptionResnetV1
from torch import Tensor, nn
from utils import CustomDataParallel, tensor_normalize

from models.compress.model import CompressModel
from models.recon.deeprecon import DeepRecon


class Model:
    def __init__(self):
        '''初始化模型'''

        # 特征提取的 faceNet
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

        # 深度重建的 base layer
        recon_net = DeepRecon().eval().cuda()
        recon_net = nn.DataParallel(recon_net).cuda()
        recon_param = torch.load(conf.recon_param_path, map_location='cuda:0')
        recon_net.load_state_dict(recon_param)
        self.recon_net = recon_net

        # 纹理增强的 enhancement layer
        e_layer = CompressModel().eval().cuda()
        e_layer = CustomDataParallel(e_layer).cuda()
        e_param = torch.load(conf.e_param_path, map_location='cuda:0')
        e_layer.load_state_dict(e_param)
        self.e_layer = e_layer

    def feat_extract(self, img: Tensor) -> Tensor:
        '''特征提取'''

        feat = self.resnet(img)
        feat = torch.squeeze(feat, 1)
        feat = torch.unsqueeze(feat, 2)
        feat = torch.unsqueeze(feat, 3)
        feat = feat.cuda()
        return feat

    def encode(self, input: Tensor) -> dict:
        '''人脸图像编码'''

        with torch.no_grad():
            # 特征提取
            feat = self.feat_extract(input)

            # 特征重建
            recon = self.recon_net(feat)

            # 残差图
            resi = (input - recon).cuda()

            # 残差图压缩
            resi_norm, intervals = tensor_normalize(resi)
            resi_norm = resi_norm.cuda()
            tex = self.e_layer.compress(resi_norm)

        return {
            'feat': feat,
            'recon': recon,
            'tex': tex,
            'intervals': intervals,
        }

    def decode(self, feat: Tensor, tex: Tensor, intervals: Tensor,
               recon: Tensor or None) -> dict:
        '''人脸图像解码'''

        with torch.no_grad():
            # 特征重建
            if recon is None:
                recon = self.recon_net(feat)

            # 纹理压缩数据解压
            resi_decoded_norm = self.e_layer.decompress(tex)
            resi_decoded = tensor_normalize(resi_decoded_norm, intervals,
                                            'anti')

        return {
            'recon': recon,
            'resi_decoded': resi_decoded,
            'resi_decoded_norm': resi_decoded_norm,
        }

    def encode_decode(self, input: Tensor) -> dict:
        '''人脸图像编码和解码，用于模型演示'''

        e = self.encode(input)
        d = self.decode(feat=e['feat'],
                        tex=e['tex'],
                        intervals=e['intervals'],
                        recon=e['recon'])

        return {
            'feat': e['feat'],
            'recon': e['recon'],
            'resi': d['resi'],
            'resi_norm': d['resi_norm'],
            'tex': e['tex'],
            'resi_decoded': d['resi_decoded'],
            'resi_decoded_norm': d['resi_decoded_norm'],
            'output': d['output'],
            'intervals': e['intervals'],
        }
