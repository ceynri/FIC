<div style="text-align: center">
<h1>FIC - Facial Image Compression Application</h1>
<div>基于深度特征辅助的人脸图像压缩系统应用</div>
</div>
<br/>

![首页.jpeg](https://i.loli.net/2021/04/25/yLnx6zAdNuQ12Vv.jpg)

针对需要同时存储人脸特征与人脸原图的应用场景，本项目设计了一套基于基于深度特征辅助的人脸图像压缩系统，并取得了优于 JPEG 压缩原图的效果。

算法模型架构基于 Shurun Wang 等人于 2019 IEEE 提出的论文 [Scalable Facial Image Compression with Deep Feature Reconstruction](https://arxiv.org/abs/1903.05921v1) 进行复现。

![SFIC.png](https://i.loli.net/2021/04/25/QxH8FuXUNclDSWG.png)

本系统基于该算法模型搭建一整套前后端应用，具备通过互联网提供算法服务的能力。

## 项目结构

- algorithm - 基于 Pytorch 的算法模块
- backend - 基于 Flask 的后端模块
- frontend - 基于 Vue 的前端模块

![人脸图像压缩系统.png](https://i.loli.net/2021/04/25/Q7RGkOuJgHcrXK6.png)

> 子模块结构与文件的对应关系请参考各个子模块根目录下的 README 文件

另外需要 Nginx 进行静态资源代理以及后端服务的反向代理实现部署

## 功能实现

- 人脸图像自动裁切归一化
- 人脸特征提取
- 人脸图像压缩
- 人脸图像解压
- 算法效果对比（与 JPEG）
- 支持多种压缩等级
- 重建效果预览
- 算法演算过程可视化
- 图像相关指标评价

## 效果预览

### 算法效果

人脸重建效果：

![人脸重建.png](https://i.loli.net/2021/04/25/xYVXzgZrqfk7Fod.png)

压缩过程例图：

![压缩过程与效果样例.jpg](https://i.loli.net/2021/04/25/UTsaFg2ZMqD9emh.jpg)

深度学习方法与 JPEG 方法在相近 bpp 条件下的 PSNR 比较：

![lambda对比.jpg](https://i.loli.net/2021/04/25/2xAvZW5UCQJt1hI.jpg)

![bpp-psnr.png](https://i.loli.net/2021/04/25/qjHrkiDsOGTmBAd.png)

### 前端交互演示

效果演示页：文件上传 -> 选择不同的压缩参数 -> 查看结果

![Demo.gif](https://i.loli.net/2021/04/25/JLnKvudAbNj8qc3.gif)

图像压缩页：文件上传 -> 下载结果

![Compress.gif](https://i.loli.net/2021/04/25/zisRAS1b7JnahHy.gif)

图像解压页：上传压缩文件 -> 下载还原图

![Decompress.gif](https://i.loli.net/2021/04/25/ndytTNa5xvDJosL.gif)
