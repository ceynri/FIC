import math
from os import path
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def psnr(img1, img2):
    '''
    使用ski_image库的psnr算法
    '''
    return peak_signal_noise_ratio(img1, img2, data_range=255)


def my_psnr(img1, img2):
    # 直接相减，求差值
    diff = img1 - img2
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    mse = np.mean(diff**2.)
    if mse == 0:
        return -1
    return 10 * math.log10(65025.0 / mse)


def ssim(img1, img2):
    '''
    使用ski_image库的ssim算法。
    在实际应用中，一般采用高斯权重计算图像的均值、方差以及协方差，而不是采用遍历像素点的方式，以换来更高的效率
    '''
    return structural_similarity(img1, img2, multichannel=True)


def my_ssim(img1, img2):
    '''实现ssim算法'''
    assert img1.shape == img2.shape
    if len(img1.shape) == 3:
        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        (img1_r, img1_g, img1_b) = cv2.split(img1)
        (img2_r, img2_g, img2_b) = cv2.split(img2)
        r_ssim = ssim(img1_r, img2_r)
        g_ssim = ssim(img1_g, img2_g)
        b_ssim = ssim(img1_b, img2_b)
        return (r_ssim + g_ssim + b_ssim) / 3
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1 = np.sqrt(((img1 - mu1)**2).mean())
    sigma2 = np.sqrt(((img2 - mu2)**2).mean())
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1 * L)**2
    C2 = (k2 * L)**2
    C3 = C2 / 2
    l12 = (2 * mu1 * mu2 + C1) / (mu1**2 + mu2**2 + C1)
    c12 = (2 * sigma1 * sigma2 + C2) / (sigma1**2 + sigma2**2 + C2)
    s12 = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    ssim_val = l12 * c12 * s12
    return ssim_val


if __name__ == "__main__":
    sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
    from utils import load_image_array

    input = load_image_array(sys.argv[1])
    output = load_image_array(sys.argv[2])
    print(psnr(input, output))
    print(ssim(input, output))
