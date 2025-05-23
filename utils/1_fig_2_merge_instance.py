import os
import numpy as np
from PIL import Image
import re


def load_image(image_path):
    """Load an image and return it as a NumPy array."""
    image = Image.open(image_path)
    return np.array(image)


def save_image(image, output_path):
    """Save a NumPy array as an image."""
    image = Image.fromarray(image)
    image.save(output_path)


def find_max_image_size(image_files):
    """Find the maximum row and column numbers from the image filenames."""
    max_row = max_col = 0
    for image_file in image_files:
        match = re.match(r"(\d+)_(\d+).tif", image_file)
        if match:
            row, col = map(int, match.groups())
            max_row = max(max_row, row)
            max_col = max(max_col, col)
    return max_row + 1, max_col + 1


def find_overlap_color(image1, image2, overlap_size, direction):
    """Find the colors in the overlap region between two images."""
    if direction == "hor12":
        overlap_colors_image1 = image1[:, -overlap_size:]
        overlap_colors_image2 = image2[:, :overlap_size]
    elif direction == "ver12":
        overlap_colors_image1 = image1[-overlap_size:, :]
        overlap_colors_image2 = image2[:overlap_size, :]
    return overlap_colors_image1, overlap_colors_image2


def spread_instances(image1, image2, overlap_colors_image1, overlap_colors_image2, overlap_size, threshold=0.9):
    """Spread instances from image1 to image2 based on overlap colors."""
    # 在图像1的重叠区域中的唯一颜色上迭代
    unique_colors_image1 = np.unique(overlap_colors_image1.reshape(-1, 3), axis=0)
    for color in unique_colors_image1:
        # 图像1重叠区域中颜色的mask
        mask_image1 = np.all(overlap_colors_image1 == color, axis=-1)

        # 在图像2的重叠区域中查找与mask_image1相对应的主色
        masked_overlap_colors_image2 = overlap_colors_image2[mask_image1]
        unique_colors_image2, counts_image2 = np.unique(masked_overlap_colors_image2.reshape(-1, 3), axis=0,
                                                        return_counts=True)
        dominant_color_index = np.argmax(counts_image2)
        dominant_color = unique_colors_image2[dominant_color_index]

        # 计算图像2重叠区域中主色的百分比
        dominant_color_count = counts_image2[dominant_color_index]
        total_pixels = np.sum(mask_image1)
        match_percentage = dominant_color_count / total_pixels

        # 如果匹配百分比超过阈值，将实例扩展到图像2（颜色变化）
        if match_percentage > threshold:
            mask_image2 = np.all(image2 == dominant_color, axis=-1)
            image2[mask_image2] = color

    return image2


def spread_instances_in_folder(folder_path, overlap_size, threshold=0.7):
    """Spread instances between adjacent images in a folder."""
    image_files = sorted(os.listdir(folder_path))
    max_row, max_col = find_max_image_size(image_files)

    for i in range(max_row):
        for j in range(max_col):
            image1_path = os.path.join(folder_path, f"{i}_{j}.tif")
            print(image1_path)
            image1 = load_image(image1_path)

            if j != max_col - 1:
                image2_path = os.path.join(folder_path, f"{i}_{j + 1}.tif")
                image2 = load_image(image2_path)
                # 横向扩散
                overlap_colors_image1, overlap_colors_image2 = find_overlap_color(image1, image2, overlap_size, "hor12")
                image2 = spread_instances(image1, image2, overlap_colors_image1, overlap_colors_image2, overlap_size, threshold)
                save_image(image2, image2_path)
            if i != max_row - 1:
                image3_path = os.path.join(folder_path, f"{i + 1}_{j}.tif")
                image3 = load_image(image3_path)
                # 纵向扩散
                overlap_colors_image1, overlap_colors_image3 = find_overlap_color(image1, image3, overlap_size, "ver12")
                image3 = spread_instances(image1, image3, overlap_colors_image1, overlap_colors_image3, overlap_size, threshold)
                save_image(image3, image3_path)

    for i in range(max_row - 2, -1, -1):
        for j in range(max_col - 2, -1, -1):
            image1_path = os.path.join(folder_path, f"{i}_{j}.tif")
            image2_path = os.path.join(folder_path, f"{i}_{j + 1}.tif")
            image3_path = os.path.join(folder_path, f"{i + 1}_{j}.tif")
            print(image1_path, image2_path, image3_path)

            image1 = load_image(image1_path)
            image2 = load_image(image2_path)
            image3 = load_image(image3_path)

            overlap_colors_image1, overlap_colors_image2 = find_overlap_color(image1, image2, overlap_size, "hor12")
            image1 = spread_instances(image2, image1, overlap_colors_image2, overlap_colors_image1, overlap_size, threshold)
            save_image(image1, image1_path)

            overlap_colors_image1, overlap_colors_image3 = find_overlap_color(image1, image3, overlap_size, "ver12")
            image1 = spread_instances(image3, image1, overlap_colors_image3, overlap_colors_image1, overlap_size, threshold)
            save_image(image1, image1_path)


# 对目录下的分割子图进行实例扩散
input_folder = "./polygon/1024_128/cropped_1024_128_g2"
overlap_size = 128
spread_instances_in_folder(input_folder, overlap_size)
