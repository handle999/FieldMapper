import argparse
import cv2
import numpy as np
import os
import sys
import torch
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# 设置环境变量 KMP_DUPLICATE_LIB_OK 为 TRUE
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def show_anns_PIL(anns):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # 获取图像的宽高
    width = sorted_anns[0]['segmentation'].shape[1]
    height = sorted_anns[0]['segmentation'].shape[0]

    # 创建一个黑色背景的图像，RGBA模式
    img = Image.new('RGBA', (width, height), (0, 0, 0, 255))

    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = tuple(int(c * 255) for c in np.random.random(3)) + (255,)  # 不透明的颜色值
        for i in range(width):
            for j in range(height):
                if m[j, i]:  # PIL中坐标为 (x, y)，与numpy数组中的索引顺序相反
                    img.putpixel((i, j), color_mask)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation script.")
    parser.add_argument("--image_path", type=str, default="input", help="Path to input images directory.")
    parser.add_argument("--image_name", type=str, default="0_0.tif", help="Name of the input image file.")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save output images directory.")
    parser.add_argument("--sam_checkpoint", type=str, default="./models/sam_vit_b_01ec64.pth",
                        help="Path to SAM model checkpoint.")
    parser.add_argument("--model_type", type=str, default="vit_b", help="Type of SAM model.")
    parser.add_argument("--min_area", type=int, default=2000, help="Min size of each instance.")
    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")
    parser.add_argument("--stability_score_thresh", type=float, default=0.92, help="a.")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88, help="b.")
    args = parser.parse_args()

    # 设置环境变量 KMP_DUPLICATE_LIB_OK 为 TRUE
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    image_path = args.image_path
    image_name = args.image_name
    output_path = args.output_path
    sam_checkpoint = args.sam_checkpoint
    model_type = args.model_type
    min_area = args.min_area

    image = cv2.imread(os.path.join(image_path, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sys.path.append("..")

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        min_mask_region_area=min_area,  # Requires open-cv to run post-processing
    )

    masks = mask_generator.generate(image)

    img = show_anns_PIL(masks)
    out_img_path = os.path.join(output_path, image_name)
    img.save(out_img_path)
