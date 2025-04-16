import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
import shutil
import os


def CollateDataset(image_dir,label_dir):  # image_dir:图片路径  label_dir：标签路径
    # 创建一个空列表来存储有效图片的路径
    valid_images = []
    # 创建一个空列表来存储有效 label 的路径
    valid_labels = []
    # 遍历 images 文件夹下的所有图片
    for image_name in os.listdir(image_dir):
        # 获取图片的完整路径
        image_path = os.path.join(image_dir, image_name)
        # 获取图片文件的扩展名
        ext = os.path.splitext(image_name)[-1]
        # 根据扩展名替换成对应的 label 文件名
        label_name = image_name.replace(ext, ".txt")
        # 获取对应 label 的完整路径
        label_path = os.path.join(label_dir, label_name)
        # 判断 label 是否存在
        if not os.path.exists(label_path):
            # # 删除图片
            # os.remove(image_path)
            print("there is no:", label_path)
        else:
            # 将图片路径添加到列表中
            valid_images.append(image_path)
            # 将 label 路径添加到列表中
            valid_labels.append(label_path)
    # 遍历每个有效图片路径
    for i in tqdm(range(len(valid_images))):
        image_path = valid_images[i]
        label_path = valid_labels[i]
        # 随机生成一个概率
        r = random.random()
        # 判断图片应该移动到哪个文件夹
        # train：valid：test = 8:2:0
        if r < 0.0:
            # 移动到 test 文件夹
            destination = "./MTdataset/test"
        elif r < 0.2:
            # 移动到 valid 文件夹
            destination = "./MTdataset/valid"
        else:
            # 移动到 train 文件夹
            destination = "./MTdataset/train"
        # 创建目标文件夹中 images 和 labels 子文件夹
        os.makedirs(os.path.join(destination, "images"), exist_ok=True)
        os.makedirs(os.path.join(destination, "labels"), exist_ok=True)
        # 生成目标文件夹中图片的新路径
        image_destination_path = os.path.join(destination, "images", os.path.basename(image_path))
        # 移动图片到目标文件夹
        shutil.copy(image_path, image_destination_path)
        # 生成目标文件夹中 label 的新路径
        label_destination_path = os.path.join(destination, "labels", os.path.basename(label_path))
        # 移动 label 到目标文件夹
        shutil.copy(label_path, label_destination_path)

if __name__ == '__main__':
    CollateDataset("./images","./labels")