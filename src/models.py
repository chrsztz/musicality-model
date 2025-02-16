# src/models.py

import torch
import torch.nn as nn


class PC_FCN(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super(PC_FCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=7, stride=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=4)

        self.conv2 = nn.Conv1d(32, 64, 7, 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=4)

        self.conv3 = nn.Conv1d(64, 128, 7, 3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=4)

        self.conv4 = nn.Conv1d(128, 16, 35, 1)
        self.bn4 = nn.BatchNorm1d(16)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x: [batch_size, input_size, seq_length]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class M_CRNN(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(M_CRNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, stride=3)
        self.bn1 = nn.BatchNorm2d(128)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=5)

        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=5)

        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=3)
        self.bn3 = nn.BatchNorm2d(512)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=5)

        self.gru = nn.GRU(input_size=512, hidden_size=200, num_layers=1, batch_first=True)
        self.fc = nn.Linear(200, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, 1, mel_bins, time_frames]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.pool3(x)

        # Reshape for GRU: [batch_size, time_steps, features]
        batch_size, channels, mel_bins, time_frames = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch_size, time_frames, channels, mel_bins]
        x = x.view(batch_size, time_frames, -1)  # [batch_size, time_frames, channels*mel_bins]

        self.gru.flatten_parameters()
        x, _ = self.gru(x)  # [batch_size, time_frames, hidden_size]
        x = x[:, -1, :]  # [batch_size, hidden_size]
        x = self.fc(x)  # [batch_size, num_classes]
        x = self.relu(x)
        return x


class PCM_CRNN(nn.Module):
    def __init__(self, pc_input_size, mel_input_channels, num_classes=1):
        super(PCM_CRNN, self).__init__()
        # PC-FCN part
        self.pc_fcn = PC_FCN(input_size=pc_input_size, num_classes=16)

        # M-CRNN part
        self.m_crnn = M_CRNN(input_channels=mel_input_channels, num_classes=200)

        # Combined part
        self.fc = nn.Linear(16 + 200, num_classes)
        self.relu = nn.ReLU()

    def forward(self, pc, mel):
        # pc: [batch_size, pc_input_size, seq_length]
        # mel: [batch_size, 1, mel_bins, time_frames]
        pc_out = self.pc_fcn(pc)  # [batch_size, 1]
        m_crnn_out = self.m_crnn(mel)  # [batch_size, num_classes]

        x = torch.cat((pc_out, m_crnn_out), dim=1)  # [batch_size, 1 + num_classes]
        x = self.fc(x)
        x = self.relu(x)
        return x
