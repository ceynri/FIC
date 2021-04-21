import config as conf
import torch
from facenet_pytorch import InceptionResnetV1
from torch import Tensor, nn
from utils import CustomDataParallel, tensor_normalize

from models.base.deeprecon import DeepRecon
from models.enhancement.gdnmodel import CompressModel


class Model:
    def __init__(self):
        '''初始化模型'''

        # base layer

        # 特征提取模型
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

        # 深度重建模型
        recon_net = DeepRecon().eval().cuda()
        recon_net = nn.DataParallel(recon_net).cuda()
        recon_param = torch.load(conf.RECON_PARAM_PATH, map_location='cuda:0')
        recon_net.load_state_dict(recon_param)
        self.recon_net = recon_net

        # enhancement layer

        # 纹理压缩模型
        e_layer = CompressModel().eval().cuda()
        e_layer = CustomDataParallel(e_layer).cuda()
        e_param = torch.load(conf.E_LAYER_PARAM_PATH, map_location='cuda:0')
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
            'input': input,
            'feat': feat,
            'recon': recon,
            'resi': resi,
            'resi_norm': resi_norm,
            'tex': tex,
            'intervals': intervals,
        }

    def decode(self,
               feat: Tensor,
               tex: Tensor,
               intervals: Tensor,
               recon: Tensor or None = None) -> dict:
        '''人脸图像解码'''

        with torch.no_grad():
            # 特征重建
            if recon is None:
                recon = self.recon_net(feat)

            # 纹理压缩数据解压
            resi_decoded_norm = self.e_layer.decompress(tex)
            resi_decoded = tensor_normalize(resi_decoded_norm, intervals,
                                            'anti')
            output = recon + resi_decoded

        return {
            'recon': recon,
            'resi_decoded': resi_decoded,
            'resi_decoded_norm': resi_decoded_norm,
            'output': output,
        }
