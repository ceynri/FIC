# 参数文件路径
RECON_PARAM_PATH = './params/recon/30w/baseLayer_7.pth'
E_LAYER_PARAM_PATH = './params/e_layer/5120/gdnmodel_10.pth'

# 图片保存位置
BASE_PATH = './public/result'
# 图片保存位置对应的URL路径
BASE_URL = '/assets/result/'

# 输入图像的尺寸
IMAGE_SIZE = 256
# 输入图像的形状
IMAGE_SHAPE = (IMAGE_SIZE, IMAGE_SIZE)
# 输入图像的像素点个数
IMAGE_PIXEL_NUM = IMAGE_SIZE * IMAGE_SIZE

# enhancement layer 参数文件路径映射表
E_PARAM_MAP = {
    'low': './params/e_layer/1024/gdnmodel_10.pth',
    'medium': './params/e_layer/5120/gdnmodel_5.pth',
    # 'medium': './params/e_layer/2560/gdnmodel_10.pth',
    'high': './params/e_layer/5120/gdnmodel_10.pth',
}
