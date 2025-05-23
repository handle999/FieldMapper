import os
import numpy as np
from PIL import Image
from skimage.io import imread
from sklearn.metrics import confusion_matrix


# 计算二分类指标
def calculate_metrics(pred_folder, lab_folder):
	# 获取所有图像文件的名称
	pred_files = sorted(os.listdir(pred_folder))
	lab_files = sorted(os.listdir(lab_folder))

	# 确保两个文件夹的文件数量一致
	assert len(pred_files) == len(lab_files), "Number of files in pred and lab folder must match"

	# 存储指标
	pixel_accuracy = 0
	mean_iou = 0
	dice_score = 0
	precision = 0
	recall = 0
	f1_score = 0
	specificity = 0

	# 累加所有文件的指标
	for pred_file, lab_file in zip(pred_files, lab_files):
		# 读取图像
		pred_path = os.path.join(pred_folder, pred_file)
		lab_path = os.path.join(lab_folder, lab_file)
		pred_img = Image.open(pred_path)  # 假设PNG文件是0和1的二值图像
		lab_img = Image.open(lab_path)
		pred_img = np.array(pred_img)
		###########################################
		# 专为0或255准备
		pred_img = (pred_img == 255).astype(int)
		###########################################
		lab_img = np.array(lab_img)
		print(pred_img)
		print(lab_img)

		# # 确保读取的图像是单通道
		# assert pred_img.ndim == 2, f"Prediction image {pred_file} is not single channel"
		# assert lab_img.ndim == 2, f"Label image {lab_file} is not single channel"

		# 计算混淆矩阵
		cm = confusion_matrix(lab_img.flatten(), pred_img.flatten(), labels=[0, 1])
		print(cm)
		TP, FN, FP, TN = cm.ravel()

		# 计算各项指标
		# For pixel_accuracy
		pixel_accuracy += (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) != 0 else 0
		# For mean_iou
		mean_iou += TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
		# For dice_score
		dice_score += 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
		# For precision
		precision += TP / (TP + FP) if (TP + FP) != 0 else 0
		# For recall
		recall += TP / (TP + FN) if (TP + FN) != 0 else 0
		# For specificity
		specificity += TN / (TN + FP) if (TN + FP) != 0 else 0

	# 计算平均指标
	num_files = len(pred_files)
	pixel_accuracy /= num_files
	mean_iou /= num_files
	dice_score /= num_files
	precision /= num_files
	recall /= num_files
	f1_score = 2 * (precision * recall) / (precision + recall)
	specificity /= num_files

	# 返回所有计算的指标
	return {
		'Pixel Accuracy': pixel_accuracy,
		'Mean IoU': mean_iou,
		'Dice Score': dice_score,
		'Precision': precision,
		'Recall': recall,
		'F1 Score': f1_score,
		'Specificity': specificity
	}


def save_metrics_to_txt(metrics, output_file):
	"""
	将指标字典保存到指定的文本文件中，每个键值对使用换行和Tab分隔。

	参数：
	metrics (dict): 包含指标的字典。
	output_file (str): 保存结果的文本文件路径。
	"""
	with open(output_file, 'w') as f:
		for metric, value in metrics.items():
			f.write(f"{metric}\t{value:.4f}\n")  # 使用Tab分隔，保留四位小数

	print(f"Metrics have been saved to {output_file}")


if __name__ == "__main__":
	model_name = "SAMRS"
	pred_folder = "./data/CROP/pred_{}".format(model_name)  # 预测文件夹路径
	lab_folder = "./data/CROP/lab"  # 标签文件夹路径
	output_file = "segmentation_metrics_{}.txt".format(model_name)  # 结果保存的文本文件路径
	os.makedirs(lab_folder, exist_ok=True)

	metrics = calculate_metrics(pred_folder, lab_folder)
	for metric, value in metrics.items():
		print(f"{metric}: {value:.4f}")
	save_metrics_to_txt(metrics, output_file)
