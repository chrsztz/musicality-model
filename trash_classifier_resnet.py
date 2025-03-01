import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# 定义数据集路径（请替换为你的实际路径）
dataset_path = 'augmented_waste_classification'

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
# 加载预训练的 ResNet-50
model = models.resnet50(pretrained=True)

# 冻结所有层（可选，如果需要微调整个网络，可以跳过此步）
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层全连接层
num_classes = len(dataset.classes)  # 自动获取类别数
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 将模型移动到设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # 支持 macOS MPS 或 CPU
model = model.to(device)
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # 只优化最后一层（如果冻结了其他层）
# 早停参数
patience = 5
best_val_loss = float('inf')
counter = 0

# TensorBoard 可视化
writer = SummaryWriter()

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 计算训练集平均损失
    train_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)

    # 验证阶段
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # 早停逻辑
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
    else:
        counter += 1
        if counter >= patience:
            print("早停触发，训练停止")
            break

writer.close()