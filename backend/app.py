from os import path

import torch
from flask import Flask, jsonify, request
# from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from torchvision.utils import save_image

import config as conf
from models.model import Model
from utils import load_image_array, tensor_to_array
from utils.eval import psnr, ssim
from utils.file import File
from utils.jpeg import dichotomy_compress

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def get_url(filename):
    return path.join(conf.BASE_URL, filename)


def get_path(filename):
    return path.join(conf.BASE_PATH, filename)


# 模型初始化
model = Model()


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

    feature_model = request.form['feature_model']
    quality_level = request.form['quality_level']
    if model.quality_level != quality_level:
        model.switch_quality_level(quality_level)
    # 将二进制转为tensor
    input = file.load_tensor().cuda()

    # 输入模型，得到返回结果
    e_data = model.encode(input)
    d_data = model.decode(feat=e_data['feat'],
                          tex=e_data['tex'],
                          intervals=e_data['intervals'],
                          recon=e_data['recon'])
    data = {**e_data, **d_data}

    # 保存压缩数据
    fic_path = get_path(f'{file.name}.fic')
    File.save_binary(
        {
            'feat': data['feat'],
            'tex': data['tex'],
            'intervals': data['intervals'],
            'ext': file.ext,
        }, fic_path)
    # fic 相关参数
    fic_size = path.getsize(fic_path)
    fic_bpp = fic_size / conf.IMAGE_PIXEL_NUM

    # 单独保存特征以计算特征和纹理的大小
    feat_path = get_path(f'{file.name}_feat.fic')
    File.save_binary({
        'feat': data['feat'],
    }, feat_path)
    # 特征相关参数
    feat_size = path.getsize(feat_path)
    feat_bpp = feat_size / conf.IMAGE_PIXEL_NUM
    # 纹理相关参数
    tex_size = fic_size - feat_size
    tex_bpp = tex_size / conf.IMAGE_PIXEL_NUM

    # 待保存图片 # TODO RENAME
    imgs = {
        'input': data['input'],
        'feat': data['recon'],
        'tex': data['resi'],
        'tex_decoded': data['resi_decoded'],
        'tex_norm': data['resi_norm'],
        'tex_decoded_norm': data['resi_decoded_norm'],
        'output': data['output'],
    }

    # 将 imgs 保存并获得对应URL
    img_urls = {}
    for key, value in imgs.items():
        # 保存图片
        file_name = file.name_suffix(key, ext='.bmp')
        file_path = get_path(file_name)
        save_image(value, file_path)
        # 返回图片url链接
        img_urls[key] = get_url(file_name)

    # 计算压缩率
    input_name = file.name_suffix('input', ext='.bmp')
    input_path = get_path(input_name)
    input_size = path.getsize(input_path)
    fic_compression_ratio = fic_size / input_size

    # jpeg对照组处理
    jpeg_name = file.name_suffix('jpeg', ext='.jpg')
    jpeg_path = get_path(jpeg_name)
    dichotomy_compress(input_path, jpeg_path, target_size=tex_size)
    img_urls['jpeg'] = get_url(jpeg_name)

    # jpeg 相关参数计算
    jpeg_size = path.getsize(jpeg_path)
    jpeg_compression_ratio = jpeg_size / input_size
    jpeg_bpp = jpeg_size / conf.IMAGE_PIXEL_NUM

    # 其他数据
    input_arr = tensor_to_array(data['input'])
    output_arr = tensor_to_array(data['output'])
    jpeg_arr = load_image_array(jpeg_path)

    # 返回的对象
    ret = {
        'image': img_urls,
        'data': get_url(f'{file.name}.fic'),
        'eval': {
            'fic_bpp': fic_bpp,
            'feat_bpp': feat_bpp,
            'tex_bpp': tex_bpp,
            'jpeg_bpp': jpeg_bpp,
            'fic_compression_ratio': fic_compression_ratio,
            'jpeg_compression_ratio': jpeg_compression_ratio,
            'fic_psnr': psnr(input_arr, output_arr),
            'fic_ssim': ssim(input_arr, output_arr),
            'jpeg_psnr': psnr(input_arr, jpeg_arr),
            'jpeg_ssim': ssim(input_arr, jpeg_arr),
        },
        'size': {
            'fic': fic_size,
            'input': input_size,
            'output': fic_size,
            'feat': feat_size,
            'tex': tex_size,
            'jpeg': jpeg_size,
        }
    }
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
        # 将二进制转为tensor
        input = file.load_tensor().cuda()
        with torch.no_grad():
            data = model.encode(input)

        # 保存压缩数据
        fic_name = f'{file.name}.fic'
        fic_path = get_path(fic_name)
        File.save_binary(
            {
                'feat': data['feat'],
                'tex': data['tex'],
                'intervals': data['intervals'],
                'ext': file.ext,
            }, fic_path)
        fic_size = path.getsize(fic_path)

        # 获取原图大小
        input_path = get_path(file.name_suffix('input', ext='.bmp'))
        save_image(input, input_path)
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

        data = model.decode(feat=fic['feat'],
                            tex=fic['tex'],
                            intervals=fic['intervals'])

        # 获取完整结果图
        x_output = data['recon'] + data['resi_decoded']

        # 保存结果图片
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
    print(f'Start serving FIC at port {port}...')
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
