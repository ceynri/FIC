import config as conf
import torch
from facenet_pytorch import InceptionResnetV1
from torch import Tensor, nn
from utils import CustomDataParallel, tensor_normalize

from models.base.deeprecon import DeepRecon
from models.enhancement.gdnmodel import GdnModel


class Model:
    def __init__(self, feature_model='facenet', quality_level='high'):
        '''初始化模型'''

        # base layer
        if feature_model == 'facenet':
            # faceNet
            facenet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
            # 特征提取网络
            self.feat_net = facenet
        # TODO more feature model

        # 反卷积重建网络
        recon_net = DeepRecon().eval().cuda()
        recon_net = nn.DataParallel(recon_net).cuda()
        recon_param = torch.load(conf.RECON_PARAM_PATH, map_location='cuda:0')
        recon_net.load_state_dict(recon_param)
        # 深度重建网络
        self.recon_net = recon_net

        # enhancement layer

        # GDN图像压缩模型
        self.gdn_model_map = self.init_gdn_model_map()
        # 纹理压缩层
        self.e_layer = self.gdn_model_map[quality_level]
        self.quality_level = quality_level

    def init_gdn_model_map(self):
        gdn_model_low = GdnModel().eval().cuda()
        gdn_model_low = CustomDataParallel(gdn_model_low).cuda()
        e_param = torch.load(conf.E_PARAM_MAP['low'], map_location='cuda:0')
        gdn_model_low.load_state_dict(e_param)

        gdn_model_medium = GdnModel().eval().cuda()
        gdn_model_medium = CustomDataParallel(gdn_model_medium).cuda()
        e_param = torch.load(conf.E_PARAM_MAP['medium'], map_location='cuda:0')
        gdn_model_medium.load_state_dict(e_param)

        gdn_model_high = GdnModel().eval().cuda()
        gdn_model_high = CustomDataParallel(gdn_model_high).cuda()
        e_param = torch.load(conf.E_PARAM_MAP['high'], map_location='cuda:0')
        gdn_model_high.load_state_dict(e_param)

        gdn_model_map = {
            'low': gdn_model_low,
            'medium': gdn_model_medium,
            'high': gdn_model_high,
        }
        return gdn_model_map

    def switch_quality_level(self, quality_level: str) -> None:
        self.quality_level = quality_level
        self.e_layer = self.gdn_model_map[quality_level]

    def feat_extract(self, img: Tensor) -> Tensor:
        '''特征提取'''

        feat = self.feat_net(img)
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
