# 参数文件路径
RECON_PARAM_PATH = './params/deconv_recon/deconv_recon_30w.pth'

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
    'low': './params/gdn_model/gdn_model_1024.pth',
    'medium': './params/gdn_model/gdn_model_2560.pth',
    'high': './params/gdn_model/gdn_model_5120.pth',
}
