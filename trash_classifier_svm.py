import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义数据集路径（请替换为你的实际路径）
dataset_path = '../datasets/augmented_waste_classification'  # 示例: 'C:/Users/YourName/waste_images'

# 定义图像预处理变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),           # 将图像调整为 224x224 像素
    transforms.RandomHorizontalFlip(),       # 随机水平翻转（数据增强）
    transforms.RandomRotation(10),           # 随机旋转 ±10 度（数据增强）
    transforms.ToTensor(),                   # 转换为 PyTorch 张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 使用 ImageNet 统计数据归一化
])

# 从文件夹加载数据集
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# 将数据集分为训练集（80%）和验证集（20%）
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 检查是否有 GPU 可用
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 加载预训练的 ResNet-50 并移除最后一层
resnet50 = models.resnet50(pretrained=True)
feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])  # 去掉最后一层全连接层
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()  # 设置为评估模式

# 定义特征提取函数
def extract_features(loader, model, device):
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="提取特征"):
            images = images.to(device)
            feats = model(images).squeeze()  # 提取特征（2048 维）
            features.append(feats.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

# 提取训练集和验证集的特征
train_features, train_labels = extract_features(train_loader, feature_extractor, device)
val_features, val_labels = extract_features(val_loader, feature_extractor, device)

# 训练 SVM 分类器
svm = LinearSVC(max_iter=1000, C=1.0)  # C 参数控制正则化强度，可调整
svm.fit(train_features, train_labels)

# 在验证集上进行预测并计算准确率
val_preds = svm.predict(val_features)
val_accuracy = accuracy_score(val_labels, val_preds)
print(f'验证集准确率: {val_accuracy * 100:.2f}%')

# 使用 t-SNE 可视化特征
tsne = TSNE(n_components=2, random_state=42)
train_tsne = tsne.fit_transform(train_features)

# 绘制 t-SNE 可视化图
plt.figure(figsize=(10, 8))
for i in range(len(dataset.classes)):  # 根据类别数动态绘制
    plt.scatter(train_tsne[train_labels == i, 0], train_tsne[train_labels == i, 1], label=dataset.classes[i])
plt.legend()
plt.title("训练集特征的 t-SNE 可视化")
plt.xlabel("t-SNE 维度 1")
plt.ylabel("t-SNE 维度 2")
plt.show()

# 保存 SVM 模型（可选）
import joblib
joblib.dump(svm, 'waste_classifier_svm.pkl')
print("SVM 模型已保存为 'waste_classifier_svm.pkl'")