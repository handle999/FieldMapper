from PIL import Image
import os


def find_minimum_bounding_box(image):
    # 找到白色区域的边界
    bbox = image.getbbox()
    return bbox


def crop_image(image, bbox):
    # 根据最小外接矩形裁剪图像
    cropped_image = image.crop(bbox)
    return cropped_image


def make_square(image):
    # 将图像变为正方形
    if image.width != image.height:
        # 计算正方形边长
        size = max(image.width, image.height)

        # 创建全黑正方形图像
        square_image = Image.new("RGB", (size, size), color=(0, 0, 0))

        # 计算粘贴位置
        x_offset = (size - image.width) // 2
        y_offset = (size - image.height) // 2
        paste_box = (x_offset, y_offset)

        # 粘贴图像
        square_image.paste(image, paste_box)
        return square_image
    else:
        return image


def mask_black_area(image, mask):
    # 将mask中黑色区域对应的区域设置为黑色
    for x in range(image.width):
        for y in range(image.height):
            if mask.getpixel((x, y)) == 0:  # 黑色像素
                image.putpixel((x, y), (0, 0, 0))  # 将对应位置的像素设置为黑色


if __name__ == "__main__":
    # 利用获取的mask，通过裁切获得polygon对应图像
    mask_folder = "./polygon/1024_128/4_masks/"
    z20_path = "./data/z20-google.tif"
    crop_folder = "./polygon/1024_128/5_crop/"

    # 打开z20.tif
    z20_image = Image.open(z20_path)

    for filename in os.listdir(mask_folder):
        if filename.endswith(".png"):
            print(filename)
            mask_path = os.path.join(mask_folder, filename)
            mask_image = Image.open(mask_path)

            # 找到mask中白色区域的边界
            bbox = find_minimum_bounding_box(mask_image)

            # 裁剪z20.tif
            cropped_image = crop_image(z20_image, bbox)
            cropped_mask = crop_image(mask_image, bbox)

            # 将mask中黑色区域对应的区域设置为黑色
            mask_black_area(cropped_image, cropped_mask)

            # 将裁剪后的图像变为正方形
            cropped_image = make_square(cropped_image)

            # 保存裁剪后的图像
            cropped_path = os.path.join(crop_folder, os.path.splitext(filename)[0] + ".png")
            if not os.path.exists(crop_folder):
                os.makedirs(crop_folder)
            cropped_image.save(cropped_path)
