import sys
from PIL import Image
from os import path


def jpeg_compress(input_path, output_path='', target_size=5120, step=1, quality=80):
    """将图片压缩到指定大小"""
    file_size = path.getsize(input_path)
    if file_size <= target_size:
        return input_path
    if output_path == '':
        path_name, _ = path.splitext(input_path)
        output_path = f'{path_name}_jpeg.jpg'
    im = Image.open(input_path)
    q = quality
    while file_size > target_size:
        im.save(output_path, quality=q)
        if q - step < 0:
            break
        q -= step
        file_size = path.getsize(output_path)
    return output_path, path.getsize(output_path)


def dichotomy_compress(input_path, output_path='', target_size=5120):
    """使用二分法，将图片压缩到指定大小"""
    # 默认输出路径
    if output_path == '':
        path_name, _ = path.splitext(input_path)
        output_path = f'{path_name}_jpeg.jpg'
    # 获取输入文件信息
    im = Image.open(input_path)
    file_size = prev_size = path.getsize(input_path)
    # 变量初始化
    left, right = 0, 100
    q = prev_q = 50
    # 二分查找
    while left <= right:
        # 保存上一次的记录
        prev_q = q
        prev_size = file_size
        # 按指定质量参数进行压缩
        q = (left + right) // 2
        im.save(output_path, quality=q)
        file_size = path.getsize(output_path)
        # 比较压缩大小
        if file_size < target_size:
            left = q + 1
        elif file_size > target_size:
            right = q - 1
        else:
            break
    # 判断最后两次压缩哪次最接近目标压缩大小
    if abs(target_size - prev_size) < abs(target_size - file_size):
        im.save(output_path, quality=prev_q)
        return output_path, prev_size
    return output_path, file_size


if __name__ == '__main__':
    file_path, target_size = sys.argv[1], int(sys.argv[2])
    # output_path, actual_size = jpeg_compress(file_path, size=target_size)
    output_path, actual_size = dichotomy_compress(file_path, target_size=target_size)
    print(target_size, actual_size)
