import os
from PIL import Image


def resize_img(input_dir, output_dir, size):
    # 检查目标目录是否存在，不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历原始目录中的PNG文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # 打开原始图片
            with Image.open(os.path.join(input_dir, filename)) as img:
                # 调整大小为256x256，并转换为RGB模式
                resized_img = img.resize((size, size)).convert("RGB")
                # 保存调整大小后的图片到目标目录，使用原始文件名
                resized_img.save(os.path.join(output_dir, filename))


if __name__ == "__main__":
    # 拉伸裁切下来的每一个polygon对应区域的大小，使其能够适应CNN输入
    input_dir = "./polygon/1024_128/5_crop/"
    output_dir = "./polygon/1024_128/6_resize/"
    size = 128
    resize_img(input_dir, output_dir, size)
