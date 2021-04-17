from os import path

import torch
from facenet_pytorch import InceptionResnetV1
from flask import Flask, jsonify, request
# from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from torch import nn
from torchvision.utils import save_image

from models.compress.model import CompressModel
from models.recon.deeprecon import DeepRecon
from utils import (CustomDataParallel, load_image_array, tensor_normalize,
                   tensor_to_array)
from utils.file import File
from utils.eval import psnr, ssim
from utils.jpeg import jpeg_compress

app = Flask(__name__)

recon_param_path = './params/recon/30w/baseLayer_5.pth'
e_param_path = './params/e_layer/enhanceLayer_2.pth'

base_path = './public/result'
base_url = '/assets/result/'

IMAGE_PIXEL_NUM = 256 * 256


def get_url(filename):
    return path.join(base_url, filename)


def get_path(filename):
    return path.join(base_path, filename)


def get_models():
    '''初始化模型'''

    # 特征提取的 faceNet
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()

    # 深度重建的 base layer
    recon_net = DeepRecon().eval()
    recon_net = recon_net.cuda()
    recon_net = nn.DataParallel(recon_net).cuda()
    recon_param = torch.load(recon_param_path, map_location='cuda:0')
    recon_net.load_state_dict(recon_param)

    # 纹理增强的 enhancement layer
    e_layer = CompressModel().eval().cuda()
    e_layer = CustomDataParallel(e_layer).cuda()
    e_param = torch.load(e_param_path, map_location='cuda:0')
    e_layer.load_state_dict(e_param)
    return resnet, recon_net, e_layer


def extract_feat(img):
    # 特征提取
    feat = resnet(img)
    feat = torch.squeeze(feat, 1)
    feat = torch.unsqueeze(feat, 2)
    feat = torch.unsqueeze(feat, 3)
    feat = feat.cuda()

    # 特征重建
    x_feat = recon_net(feat)

    return feat, x_feat


# 模型初始化
resnet, recon_net, e_layer = get_models()


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
        x_input = file.load_tensor().cuda()

        # 特征提取与重建
        feat, x_feat = extract_feat(x_input)

        # 残差纹理图
        x_tex = (x_input - x_feat).cuda()

        # 残差纹理图压缩
        x_tex_norm, intervals = tensor_normalize(x_tex)
        x_tex_norm = x_tex_norm.cuda()
        tex = e_layer.compress(x_tex_norm)

        # 纹理压缩数据解压
        x_tex_decoded_norm = e_layer.decompress(tex)
        x_tex_decoded = tensor_normalize(x_tex_decoded_norm, intervals, 'anti')

        # 获取完整结果图
        x_output = x_feat + x_tex_decoded

        # 保存压缩数据
        fic_path = get_path(f'{file.name}.fic')
        File.save_binary(
            {
                'feat': feat,
                'tex': tex,
                'intervals': intervals,
                'ext': file.ext,
            }, fic_path)
        fic_size = path.getsize(fic_path)
        fic_bpp = fic_size / IMAGE_PIXEL_NUM

        feat_path = get_path(f'{file.name}_feat.fic')
        File.save_binary({
            'feat': feat,
        }, feat_path)
        feat_size = path.getsize(feat_path)
        tex_size = fic_size - feat_size
        feat_bpp = feat_size / IMAGE_PIXEL_NUM
        tex_bpp = tex_size / IMAGE_PIXEL_NUM

        # 待保存图片
        result = {
            'input': x_input,
            'feat': x_feat,
            'tex': x_tex,
            'tex_decoded': x_tex_decoded,
            'tex_norm': x_tex_norm,
            'tex_decoded_norm': x_tex_decoded_norm,
            'output': x_output,
        }

        # 其他数据
        x_input_arr = tensor_to_array(x_input)
        x_output_arr = tensor_to_array(x_output)
        ret = {
            'image': {},
            'data': get_url(f'{file.name}.fic'),
            'eval': {
                'fic_psnr': psnr(x_input_arr, x_output_arr),
                'fic_ssim': ssim(x_input_arr, x_output_arr),
                'fic_bpp': fic_bpp,
                'feat_bpp': feat_bpp,
                'tex_bpp': tex_bpp,
            },
            'size': {
                'fic': fic_size,
                'output': fic_size,
                'feat': feat_size,
                'tex': tex_size,
            }
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

        ret['size']['input'] = input_size
        ret['eval']['fic_compression_ratio'] = fic_compression_ratio

        # jpeg对照组处理
        jpeg_name = file.name_suffix('jpeg', ext='.jpg')
        jpeg_path = get_path(jpeg_name)
        jpeg_compress(input_path, jpeg_path, size=tex_size, quality=50)

        ret['image']['jpeg'] = get_url(jpeg_name)
        jpeg_arr = load_image_array(jpeg_path)
        ret['eval']['jpeg_psnr'] = psnr(x_input_arr, jpeg_arr)
        ret['eval']['jpeg_ssim'] = ssim(x_input_arr, jpeg_arr)

        jpeg_size = path.getsize(jpeg_path)
        ret['size']['jpeg'] = jpeg_size
        jpeg_compression_ratio = jpeg_size / input_size
        ret['eval']['jpeg_compression_ratio'] = jpeg_compression_ratio
        ret['eval']['jpeg_bpp'] = jpeg_size / IMAGE_PIXEL_NUM

        # 响应请求
        response = jsonify(ret)
        return response


@app.route('/compress', methods=['POST'])
def compress():
    '''批量压缩图片并返回压缩结果'''

    # 获取文件对象
    files = request.files.getlist('files')
    ret = []
    for rawfile in files:
        file = File(rawfile)
        with torch.no_grad():
            # 将二进制转为tensor
            x_input = file.load_tensor().cuda()

            # 特征提取与重建
            feat, x_feat = extract_feat(x_input)

            # 残差纹理图
            x_tex = (x_input - x_feat).cuda()

            # 残差纹理图压缩
            x_tex_norm, intervals = tensor_normalize(x_tex)
            x_tex_norm = x_tex_norm.cuda()
            tex = e_layer.compress(x_tex_norm)

        # 保存压缩数据
        fic_name = f'{file.name}.fic'
        fic_path = get_path(fic_name)
        File.save_binary(
            {
                'feat': feat,
                'tex': tex,
                'intervals': intervals,
                'ext': file.ext,
            }, fic_path)
        fic_size = path.getsize(fic_path)

        # 获取原图大小
        input_path = get_path(file.name_suffix('input', ext='.bmp'))
        save_image(x_input, input_path)
        input_size = path.getsize(input_path)
        fic_compression_ratio = fic_size / input_size

        # 待返回的结果数据
        result = {
            'name': fic_name,
            'data': get_url(fic_name),
            'size': fic_size,
            'compression_ratio': fic_compression_ratio,
        }
        ret.append(result)

    # 响应请求
    response = jsonify(ret)
    return response


@app.route('/decompress', methods=['POST'])
def decompress():
    '''批量解压fic文件并返回解压后的图片'''

    # 获取文件对象
    files = request.files.getlist('files')
    ret = []
    for rawfile in files:
        # 获取fic对象
        fic = File.load_binary(rawfile)
        file = File(rawfile)
        with torch.no_grad():

            # 特征重建
            x_feat = recon_net(fic['feat'])

            # 纹理压缩数据解压
            x_tex_decoded_norm = e_layer.decompress(fic['tex'])
            x_tex_decoded = tensor_normalize(x_tex_decoded_norm,
                                             fic['intervals'], 'anti')

        # 获取完整结果图
        x_output = x_feat + x_tex_decoded

        # file_name = file.name_suffix('fic', ext='.bmp')
        file_name = file.name_suffix('fic', ext=fic['ext'])
        file_path = get_path(file_name)
        save_image(x_output, file_path)

        # 待返回的结果数据
        result = {
            'name': file_name,
            'data': get_url(file_name),
            'size': path.getsize(file_path),
        }
        ret.append(result)
    # 响应请求
    response = jsonify(ret)
    return response


if __name__ == '__main__':
    # 监听服务端口
    port = 1127
    print(f'Start serving style transfer at port {port}...')
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
