import os
import shutil
import random

dataset_dir = "dataset-original"
train_dir = "dataset-original/train"
val_dir = "dataset-original/val"
split_ratio = 0.8  # 80%训练，20%验证

for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    if not os.path.isdir(category_path):
        continue
    images = os.listdir(category_path)
    random.shuffle(images)  # 随机打乱
    train_size = int(len(images) * split_ratio)

    # 创建训练和验证目录
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    # 划分
    for img in images[:train_size]:
        shutil.move(os.path.join(category_path, img), os.path.join(train_dir, category, img))
    for img in images[train_size:]:
        shutil.move(os.path.join(category_path, img), os.path.join(val_dir, category, img))