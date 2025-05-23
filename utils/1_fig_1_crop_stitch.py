import os
from PIL import Image
from osgeo import gdal
import math
import numpy as np


def crop_image(input_file, output_dir, crop_size=(512, 512), overlap=32):
    # 打开遥感图像
    ds = gdal.Open(input_file)
    width = ds.RasterXSize
    height = ds.RasterYSize

    # 创建输出路径
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 计算裁切数量
    rows = math.ceil((height - overlap) / (crop_size[1] - overlap))
    cols = math.ceil((width - overlap) / (crop_size[0] - overlap))
    print(cols, rows)

    # 裁切并保存
    for i in range(rows):
        for j in range(cols):
            x_offset = j * (crop_size[0] - overlap)
            y_offset = i * (crop_size[1] - overlap)
            output_file = f"{output_dir}/{i}_{j}.tif"
            gdal.Translate(output_file, ds, srcWin=[x_offset, y_offset, crop_size[0], crop_size[1]])
    ds = None


def stitch_images(input_dir, output_file, crop_size=(1024, 1024), overlap=32, init_size=(6625, 4972)):
    # 获取拼接图片列表
    image_files = sorted([file for file in os.listdir(input_dir) if file.endswith('.tif')])

    # 计算最终拼接图像的行列数
    num_rows = int(math.ceil(init_size[1] / (crop_size[1] - overlap)))
    num_cols = int(math.ceil(init_size[0] / (crop_size[0] - overlap)))

    # 计算缝合图像的尺寸
    stitched_height = num_rows * (crop_size[1] - overlap) + overlap
    stitched_width = num_cols * (crop_size[0] - overlap) + overlap

    # 空白数组，保存拼接图片
    stitched_image = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    # 遍历在每个裁剪的子图像，进行拼接
    for image_file in image_files:
        # 从文件命中获取行列
        row, col = map(int, image_file.split('.')[0].split('_'))

        # 计算在最终图中的x、y偏移量
        y_offset = row * (crop_size[1] - overlap)
        x_offset = col * (crop_size[0] - overlap)

        # 通过gdal读取tif图
        cropped_image = gdal.Open(os.path.join(input_dir, image_file))
        image_data = cropped_image.ReadAsArray()

        # 如果有4通道，去除经纬度通道（后续重新获取，防止SAM分割结果不保存经纬度信息）
        if image_data.shape[0] == 4:
            image_data = image_data[:3, :, :]

        # 计算大小并拼接
        height, width = image_data.shape[1], image_data.shape[2]
        stitched_image[y_offset:y_offset + height, x_offset:x_offset + width, :] = image_data.transpose(1, 2, 0)

    # 为了确保大小一致，由于子图会有多余的黑边，进行裁切去除
    stitched_image = stitched_image[:init_size[1], :init_size[0], :]

    # 保存tif图像
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, init_size[0], init_size[1], 3, gdal.GDT_Byte)
    out_ds.GetRasterBand(1).WriteArray(stitched_image[:, :, 0])
    out_ds.GetRasterBand(2).WriteArray(stitched_image[:, :, 1])
    out_ds.GetRasterBand(3).WriteArray(stitched_image[:, :, 2])
    out_ds.FlushCache()
    out_ds = None


# 导入图片
input_file = "./data/z20-google.tif"       # 输入图片
output_dir = "./polygon/1024_128/cropped_1024_128_g2"   # 裁切图片的存取路径
output_file = "./polygon/1024_128/1024_128_SAM_g2.tif"  # 输出拼接好的图片
cropsize = 1024             # 裁剪图片大小
overlap = 128               # 裁切重叠区域大小

crop_image(input_file, output_dir, crop_size=(cropsize, cropsize), overlap=overlap)
stitch_images(output_dir, output_file, init_size=(6625, 4972), overlap=overlap, crop_size=(cropsize, cropsize))
