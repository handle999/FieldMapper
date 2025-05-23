import os
import subprocess


def process_tif_files(input_folder, output_folder, sam_checkpoint, model_type, min_area, device):
    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    for tif_file in tif_files:
        print(tif_file)
        command = [
            "python",
            "sam_one_save.py",
            "--image_path", input_folder,
            "--image_name", tif_file,
            "--output_path", output_folder,
            "--sam_checkpoint", sam_checkpoint,
            "--model_type", model_type,
            "--min_area", str(min_area),
            "--device", device,
            "--pred_iou_thresh", 0.95,
            "--stability_score_thresh", 0.92,
        ]
        subprocess.run(command)


if __name__ == "__main__":
    input_folder = "./data/cropped_1024_128/"
    output_folder = "./polygon/cropped_1024_128/"
    sam_checkpoint = "./models/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    min_area = 2000
    device = "cuda"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    process_tif_files(input_folder, output_folder, sam_checkpoint, model_type, min_area, device)
