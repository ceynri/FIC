from os import path

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from flask import Flask, jsonify, request
# from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from torch import nn
from torchvision.utils import save_image

from DeepRcon import DRcon
from endtoend import AutoEncoder
from eval_index import psnr, ssim
from utils import (
    CustomDataParallel, File, load_image_array, save_fic, tensor_normalize,
    tensor_to_array
)

app = Flask(__name__)
# CORS(app, supports_credentials=True)

base_path = './public/result'
base_url = '/assets/result/'


def get_url(filename):
    return path.join(base_url, filename)


def get_path(filename):
    return path.join(base_path, filename)


def get_models():
    '''初始化模型'''

    # FaceNet for extracting features
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

    # reconstruct face by feature
    b_layer = DRcon().eval()
    b_layer = b_layer.cuda()
    b_layer = nn.DataParallel(b_layer).cuda()
    b_param = torch.load('./data/b_layer_fcn.pth', map_location='cuda:0')
    b_layer.load_state_dict(b_param)

    # enhancement
    e_layer = AutoEncoder().eval().cuda()
    e_layer = CustomDataParallel(e_layer).cuda()
    c_param = torch.load('./data/e_layer_5120.pth', map_location='cuda:0')
    e_layer.load_state_dict(c_param)
    return resnet, b_layer, e_layer


def extract_feat(img):
    # 特征提取
    feat = resnet(img)
    feat = torch.squeeze(feat, 1)
    feat = torch.unsqueeze(feat, 2)
    feat = torch.unsqueeze(feat, 3)
    feat = feat.cuda()

    # 特征重建
    x_feat = b_layer(feat)

    return feat, x_feat


# 模型初始化
resnet, b_layer, e_layer = get_models()


@app.route('/')
def hello():
    '''测试服务接口联通性'''

    return "Hello FIC!"


@app.route('/demo_process', methods=['POST'])
def demo_process():
    '''提供demo展示功能'''

    # 获取文件对象
    file = request.files['file']
    file = File(file)

    with torch.no_grad():
        # 将二进制转为tensor
        x_input = file.load_tensor()
        x_input = x_input.cuda()

        # 特征提取与重建
        feat, x_feat = extract_feat(x_input)

        # 残差纹理图
        x_resi = x_input - x_feat
        x_resi = x_resi.cuda()

        # 残差纹理图压缩
        x_resi_norm, intervals = tensor_normalize(x_resi)
        x_resi_norm = x_resi_norm.cuda()
        tex = e_layer.compress(x_resi_norm)

        # 纹理压缩数据解压
        x_recon_norm = e_layer.decompress(tex)
        x_recon = tensor_normalize(x_recon_norm, intervals, 'anti')

        # 获取完整结果图
        x_output = x_feat + x_recon

        # 保存压缩数据
        fic_path = get_path(f'{file.name}.fic')
        save_fic(feat, tex, fic_path)
        fic_size = path.getsize(fic_path)

        # 待保存图片
        result = {
            'input': x_input,
            'feat': x_feat,
            'resi': x_resi,
            'recon': x_recon,
            'resi_norm': x_resi_norm,
            'recon_norm': x_recon_norm,
            'output': x_output,
        }

        # 其他数据
        x_input_arr = tensor_to_array(x_input)
        x_output_arr = tensor_to_array(x_output)
        ret = {
            'image': {},
            'data': get_url(f'{file.name}.fic'),
            'size': {},
            'eval': {
                'fic_psnr': psnr(x_input_arr, x_output_arr),
                'fic_ssim': ssim(x_input_arr, x_output_arr),
            },
        }
        for key, value in result.items():
            # 保存图片
            file_name = file.name_suffix(key, ext='.bmp')
            file_path = get_path(file_name)
            save_image(value, file_path)
            # 返回图片url链接
            ret['image'][key] = get_url(file_name)

        # 计算压缩率
        input_path = get_path(file.name_suffix('input', ext='.bmp'))
        input_size = path.getsize(input_path)
        fic_compression_ratio = fic_size / input_size

        ret['size'] = {
            'fic': fic_size,
            'output': fic_size,
            'input': input_size,
        }
        ret['eval']['fic_compression_ratio'] = fic_compression_ratio

        # jpeg（临时）
        jpeg_name = file.name_suffix('jpeg', ext='.jpg')
        jpeg_path = get_path(jpeg_name)
        save_image(x_input, jpeg_path)

        ret['image']['jpeg'] = get_url(jpeg_name)
        jpeg_arr = load_image_array(jpeg_path)
        ret['eval']['jpeg_psnr'] = psnr(x_input_arr, jpeg_arr)
        ret['eval']['jpeg_ssim'] = ssim(x_input_arr, jpeg_arr)

        jpeg_size = path.getsize(jpeg_path)
        ret['size']['jpeg'] = jpeg_size
        jpeg_compression_ratio = jpeg_size / input_size
        ret['eval']['jpeg_compression_ratio'] = jpeg_compression_ratio

        # 响应请求
        response = jsonify(ret)
        return response


# @app.route('/compress', methods=['POST'])
# def compress():
#     '''批量压缩图片并返回压缩结果'''

#     # 获取文件对象
#     files = request.files.getlist('files')
#     # TODO
#     file = File(files[0])

#     with torch.no_grad():
#         # 将二进制转为tensor
#         x_input = file.load_tensor()
#         x_input = x_input.cuda()

#         # 特征提取与重建
#         feat, x_feat = extract_feat(x_input)

#         # 残差纹理图
#         x_resi = x_input - x_feat
#         x_resi = x_resi.cuda()

#         # 残差纹理图压缩
#         x_resi_norm, intervals = tensor_normalize(x_resi)
#         x_resi_norm = x_resi_norm.cuda()
#         tex = e_layer.compress(x_resi_norm), intervals

#         # 保存压缩数据
#         save_path = get_path(f'{file.name}.fic')
#         save_fic(feat, tex, save_path)

#         # 保存图片和url路径
#         result = {
#             'input': x_input,
#         }
#         ret = {
#             'name': file.name,
#             'data': get_url(f'{file.name}.fic'),
#         }
#         for key, value in result.items():
#             filename = file.name_suffix(key)
#             save_image(value, get_path(filename))
#             ret[key] = get_url(filename)

#         # 响应请求
#         response = jsonify(ret)
#         return response

if __name__ == '__main__':
    # 监听服务端口
    port = 1127
    print(f'Start serving style transfer at port {port}...')
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
