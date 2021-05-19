# FIC 算法模块

本模块为算法模型的训练代码

> FIC 系统仅使用了 base_enhance 作为人脸重建算法主模型，基于 GAN 的人脸重建算法主要用于测试 GAN 方法在面向图像压缩的人脸重建效果。
>
> 代码测试性较强，无整理，仅供参考

## 项目结构

```
├── base_enhance                基础层与增强层的实现
│   ├── train_base.py           训练基础层
│   ├── train_enhancement.py    训练增强层
│   ├── deconv_recon.py         基础层的人脸重建网络
│   └── gdn_model.py            增强层模型
├── facenet_gan                 基于GAN的人脸重建
├── facenet_wgan                基于WGAN的人脸重建
├── resnet_gan                  基于GAN的人脸重建，使用resnet代替faceNet提取人脸特征
└── resnet_wgan                 基于WGAN的人脸重建，使用resnet代替faceNet提取人脸特征
```

## 训练环境

- Ubuntu Linux
- TITAN X (Pascal) 12GB 若干
- CUDA 10.2
- Pytorch 1.7.1

## 训练数据集

VGGFace2

# 网络结构

## 原论文框架

[![origin.png](https://z3.ax1x.com/2021/05/19/g5Hu7T.md.png)](https://imgtu.com/i/g5Hu7T)

 Shurun Wang 等人于 2019 IEEE 提出的论文 [Scalable Facial Image Compression with Deep Feature Reconstruction](https://arxiv.org/abs/1903.05921v1)

## 基于GAN的人脸重建网络

[![GAN.png](https://z3.ax1x.com/2021/05/19/g5Hcut.md.png)](https://imgtu.com/i/g5Hcut)

$$Loss=\lambda_{M}MAE(I_G,I)+\lambda_{S}SSIM(I_G,I)+\lambda_{G}L_{GAN}$$

## 改进方案



[![improved.png](https://z3.ax1x.com/2021/05/19/g5HI3j.md.png)](https://imgtu.com/i/g5HI3j)

$$Loss=\lambda_{M}MAE(I_G,I)+\lambda_{S}SSIM(I_G,I)+\lambda_{G}L_{GAN}+\lambda_PMSE(F(I),G(I))$$

## 预训练模型

见 [Github releases](https://github.com/ceynri/FIC/releases)

## 开源代码参考

- [InterDigitalInc/CompressAI](https://github.com/InterDigitalInc/CompressAI)
- [tensorflow/compression](https://github.com/tensorflow/compression)
