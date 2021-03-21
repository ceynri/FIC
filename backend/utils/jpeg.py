import sys
from PIL import Image
from os import path


def jpeg_compress(input_path, output_path='', size=5120, step=1, quality=80):
    """将图片压缩到指定大小"""
    file_size = path.getsize(input_path)
    if file_size <= size:
        return input_path
    if output_path == '':
        name, _ = path.splitext(input_path.filename)
        output_path = f'{name}_jpeg.jpg'
    q = quality
    while file_size > size:
        im = Image.open(input_path)
        im.save(output_path, quality=q)
        if q - step < 0:
            break
        q -= step
        file_size = path.getsize(output_path)
    return output_path, path.getsize(output_path)


if __name__ == '__main__':
    file_path, output_path, size = sys.argv[1], sys.argv[2], sys.argv[3]
    output_path, size = jpeg_compress(file_path, output_path, size)
    print(size)
