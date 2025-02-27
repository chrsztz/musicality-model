# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loading import PianoPerformanceDataset, PianoDataset
import pretty_midi
from tqdm import tqdm
from feature_extraction import FeatureExtractor
from models import PC_FCN, M_CRNN, PCM_CRNN
import argparse
import os


def train_model(model, dataloader, criterion, optimizer, device, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")
    return model


def main():
    parser = argparse.ArgumentParser(description='Train Piano Performance Evaluation Model')
    parser.add_argument('--midi_dir', type=str, default='data/all_2rounds', help='Path to MIDI files directory')
    parser.add_argument('--label_csv', type=str, default='data/labels/total_2rounds.csv', help='Path to label CSV file')
    parser.add_argument('--model', type=str, choices=['PC_FCN', 'M_CRNN', 'PCM_CRNN'], default='PC_FCN',
                        help='Model type to train')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='models/', help='Path to save trained model')

    args = parser.parse_args()

    # 创建保存模型的目录
    os.makedirs(args.save_path, exist_ok=True)

    # 数据加载与预处理
    dataset = PianoPerformanceDataset(args.midi_dir, args.label_csv)
    data = dataset.get_data()
    feature_extractor = FeatureExtractor()
    piano_dataset = PianoDataset(data, feature_extractor)
    dataloader = DataLoader(piano_dataset, batch_size=args.batch_size, shuffle=True)

    # 模型定义
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.model == 'PC_FCN':
        input_size = len(sorted(feature_extractor.extract_features(pretty_midi.PrettyMIDI())))
        model = PC_FCN(input_size=input_size)
    elif args.model == 'M_CRNN':
        input_channels = 1  # 假设mel spectrogram是单通道
        model = M_CRNN(input_channels=input_channels)
    elif args.model == 'PCM_CRNN':
        pc_input_size = len(sorted(feature_extractor.extract_features(pretty_midi.PrettyMIDI())))
        mel_input_channels = 1
        model = PCM_CRNN(pc_input_size=pc_input_size, mel_input_channels=mel_input_channels)

    model.to(device)

    # 损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练模型
    trained_model = train_model(model, dataloader, criterion, optimizer, device, num_epochs=args.num_epochs)

    # 保存模型
    model_save_path = os.path.join(args.save_path, f"{args.model}.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
