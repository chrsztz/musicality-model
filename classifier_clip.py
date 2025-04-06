import clip
import torch
from PIL import Image
import os

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义类别（与文件夹名称一致）
categories = ["Battery", "Biological", "Cardboard", "Clothes", "Glass","Metal","Paper","Plastic","Shoes","Trash"]  # 根据你的文件夹调整
text_inputs = clip.tokenize([f"This is a {category} object." for category in categories]).to(device)

# 数据集路径
dataset_path = "augmented_waste_classification/val"  # 替换为你的实际路径

correct = 0
total = 0

for category in categories:
    folder_path = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            # 加载并预处理图片
            image = Image.open(img_path)
            image_input = preprocess(image).unsqueeze(0).to(device)

            # 推理
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                text_features = model.encode_text(text_inputs)
                logits_per_image, _ = model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            # 预测类别
            predicted_category = categories[probs.argmax()]

            # 检查是否正确
            if predicted_category == category:
                correct += 1
            total += 1
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {e}")

# 计算并输出准确率
accuracy = correct / total if total > 0 else 0
print(f"总图片数: {total}")
print(f"正确预测数: {correct}")
print(f"准确率: {accuracy:.2%}")