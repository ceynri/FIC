from os import path

from flask import Flask, jsonify, request
from gevent.pywsgi import WSGIServer

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torchvision.utils import save_image

from DeepRcon import DRcon
from endtoend import AutoEncoder
from utils import File, save_compressed_data, tensor_normalize

app = Flask(__name__)
base_path = './public/result'


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


# FaceNet for extracting features
resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

# reconstruct face by feature
base = DRcon().eval()
base = base.cuda()
base = nn.DataParallel(base).cuda()
param = torch.load('./data/base_layer.pth')
base.load_state_dict(param)

# enhancement
codec = AutoEncoder().eval().cuda()
codec = CustomDataParallel(codec).cuda()
c_param = torch.load('./data/enhanceLayer_lamda.pth')
codec.load_state_dict(c_param)


@app.route('/')
def hello():
    return "Hello World!"


@app.route('/demo_process', methods=['POST'])
def demo_process():
    file = request.files['file']
    file = File(file)

    with torch.no_grad():
        x_input = file.load_tensor()
        x_input = x_input.cuda()

        feat = resnet(x_input)
        feat = torch.squeeze(feat, 1)
        feat = torch.unsqueeze(feat, 2)
        feat = torch.unsqueeze(feat, 3)
        feat = feat.cuda()

        x_feat = base(feat)

        x_resi = x_input - x_feat
        x_resi = x_resi.cuda()
        tex = codec.compress(x_resi)
        x_recon = codec.decompress(tex)
        x_output = x_feat + x_recon

        x_resi_norm = tensor_normalize(x_resi)
        x_recon_norm = tensor_normalize(x_recon)

        save_path_name = path.join(base_path, f'{file.name}.fic')
        save_compressed_data(feat, tex, save_path_name)

        result = {
            'input': x_input,
            'feat': x_feat,
            'resi': x_resi,
            'recon': x_recon,
            'resi_norm': x_resi_norm,
            'recon_norm': x_recon_norm,
            'output': x_output,
        }
        ret_value = {}
        for key, value in result.items():
            file_name = file.name_suffix(key)
            save_image(value, path.join(base_path, file_name))
            ret_value[key] = f'http://127.0.0.1/assets/result/{file_name}'

        response = jsonify(ret_value)
        return response


if __name__ == '__main__':
    port = 1127
    print(f'Start serving style transfer at port {port}...')
    http_server = WSGIServer(('', port), app)
    http_server.serve_forever()
