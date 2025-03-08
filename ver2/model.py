import torch
import torch.nn as nn
import torch.nn.functional as F


class PitchContourCNN(nn.Module):
    """
    基于音高轮廓的卷积神经网络模型
    类似于论文中的PC-FCN (Pitch Contour - Fully Convolutional Network)
    """

    def __init__(self):
        super(PitchContourCNN, self).__init__()

        # 第一层卷积层：输入[batch_size, 1, seq_len]
        self.conv1 = nn.Conv1d(1, 4, kernel_size=7, stride=3)
        self.bn1 = nn.BatchNorm1d(4)

        # 第二层卷积层
        self.conv2 = nn.Conv1d(4, 8, kernel_size=7, stride=3)
        self.bn2 = nn.BatchNorm1d(8)

        # 第三层卷积层
        self.conv3 = nn.Conv1d(8, 16, kernel_size=7, stride=3)
        self.bn3 = nn.BatchNorm1d(16)

        # 最后一层卷积层 - 输出单个通道
        self.conv4 = nn.Conv1d(16, 1, kernel_size=35, stride=1)

        # 平均池化用于输出单一评分
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, 1, seq_len]

        Returns:
            output: 模型输出 [batch_size, 1]
        """
        # 第一层卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # 第二层卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # 第三层卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # 最后一层卷积
        x = self.conv4(x)

        # 平均池化得到单一评分
        x = self.avg_pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        return x


class MelSpectrogramCRNN(nn.Module):
    """
    基于梅尔频谱图的卷积循环神经网络模型
    类似于论文中的M-CRNN (Mel-spectrogram - Convolutional Recurrent Neural Network)
    """

    def __init__(self, input_channels=1, hidden_size=200):
        super(MelSpectrogramCRNN, self).__init__()

        self.hidden_size = hidden_size

        # 第一层卷积块：输入[batch_size, input_channels, freq_bins, time_frames]
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(3, 7), stride=1, padding=(1, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 4))

        # 第二层卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 7), stride=1, padding=(1, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 5))

        # 第三层卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 7), stride=1, padding=(1, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 5))

        # GRU层
        self.gru = nn.GRU(
            input_size=128,  # 假设经过池化后的特征维度
            hidden_size=hidden_size,
            batch_first=True
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, input_channels, freq_bins, time_frames]

        Returns:
            output: 模型输出 [batch_size, 1]
        """
        batch_size = x.size(0)

        # 第一层卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.pool1(x)

        # 第二层卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool2(x)

        # 第三层卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool3(x)

        # 准备GRU输入 - 将频率维度作为特征
        # [batch_size, channels, freq, time] -> [batch_size, time, channels]
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, x.size(1), -1)

        # GRU层
        output, hidden = self.gru(x)

        # 使用最后一个时间步的隐藏状态
        x = hidden.view(batch_size, -1)

        # 输出层
        x = self.fc(x)

        return x


class HybridPianoModel(nn.Module):
    """
    结合音高轮廓和梅尔频谱图的混合模型
    类似于论文中的PCM-CRNN (Pitch Contour + Mel-spectrogram - CRNN)
    """

    def __init__(self, pc_cnn=None, mel_crnn=None):
        super(HybridPianoModel, self).__init__()

        # 音高轮廓CNN部分
        self.pc_cnn = pc_cnn if pc_cnn else PitchContourCNN()

        # 修改PC-CNN的最后一层，输出RNN层而不是最终得分
        if hasattr(self.pc_cnn, 'conv4'):
            # 删除最后一层卷积
            delattr(self.pc_cnn, 'conv4')
        if hasattr(self.pc_cnn, 'avg_pool'):
            # 删除平均池化层
            delattr(self.pc_cnn, 'avg_pool')

        # 为PC-CNN添加GRU层
        self.pc_rnn = nn.GRU(
            input_size=16,  # 与conv3的输出通道数匹配
            hidden_size=16,
            batch_first=True
        )

        # 梅尔频谱CRNN部分
        self.mel_crnn = mel_crnn if mel_crnn else MelSpectrogramCRNN()

        # 为Mel-CRNN输出添加降维层
        self.mel_fc = nn.Linear(200, 16)  # 200是Mel-CRNN的隐藏层大小

        # 特征合并后的输出层
        self.output_fc = nn.Linear(32, 1)  # 16(PC) + 16(Mel) = 32

    def forward(self, pc_input, mel_input):
        """
        前向传播

        Args:
            pc_input: 音高轮廓输入 [batch_size, 1, pc_seq_len]
            mel_input: 梅尔频谱图输入 [batch_size, 1, freq_bins, time_frames]

        Returns:
            output: 模型输出 [batch_size, 1]
        """
        batch_size = pc_input.size(0)

        # 处理音高轮廓
        if hasattr(self.pc_cnn, 'forward'):
            # 使用PC-CNN的前三层
            x_pc = self.pc_cnn.conv1(pc_input)
            x_pc = self.pc_cnn.bn1(x_pc)
            x_pc = F.relu(x_pc)

            x_pc = self.pc_cnn.conv2(x_pc)
            x_pc = self.pc_cnn.bn2(x_pc)
            x_pc = F.relu(x_pc)

            x_pc = self.pc_cnn.conv3(x_pc)
            x_pc = self.pc_cnn.bn3(x_pc)
            x_pc = F.relu(x_pc)
        else:
            # 直接使用pc_cnn作为结果（用于自定义PC处理器）
            x_pc = self.pc_cnn(pc_input)

        # 准备RNN输入
        x_pc = x_pc.permute(0, 2, 1)  # [batch_size, seq_len, channels]

        # GRU层
        _, h_pc = self.pc_rnn(x_pc)
        h_pc = h_pc.view(batch_size, -1)

        # 处理梅尔频谱
        x_mel = self.mel_crnn(mel_input)

        # 降维
        x_mel = self.mel_fc(x_mel)

        # 合并特征
        combined = torch.cat((h_pc, x_mel), dim=1)

        # 输出层
        output = self.output_fc(combined)

        return output


class MultiTaskPianoModel(nn.Module):
    """
    多任务学习模型，同时预测多个评估维度
    """

    def __init__(self, base_model, num_tasks=4):
        super(MultiTaskPianoModel, self).__init__()

        self.num_tasks = num_tasks
        self.base_model = base_model

        # 为每个任务创建一个输出头
        if isinstance(base_model, HybridPianoModel):
            # 混合模型 - 输入维度32
            self.task_heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(num_tasks)])
            # 替换原始输出层
            self.base_model.output_fc = nn.Identity()
        elif isinstance(base_model, MelSpectrogramCRNN):
            # CRNN模型 - 输入维度200
            self.task_heads = nn.ModuleList([nn.Linear(200, 1) for _ in range(num_tasks)])
            # 替换原始输出层
            self.base_model.fc = nn.Identity()
        elif isinstance(base_model, PitchContourCNN):
            # CNN模型 - 需要修改forward方法
            # 这里假设最后一层卷积的输出是16通道
            self.task_heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(num_tasks)])
        else:
            raise ValueError("不支持的基础模型类型")

    def forward(self, *args):
        """
        前向传播

        Args:
            *args: 基础模型的输入参数

        Returns:
            outputs: 多个任务的输出列表 [batch_size, num_tasks]
        """
        # 获取基础模型的特征表示
        features = self.base_model(*args)

        # 对每个任务应用单独的输出头
        outputs = [head(features) for head in self.task_heads]

        # 合并所有任务的输出
        return torch.cat(outputs, dim=1)


class HierarchicalAttentionNetwork(nn.Module):
    """
    分层注意力网络，用于对齐乐谱和演奏，类似PercePiano论文中提到的结构
    """

    def __init__(self, input_dim, hidden_dim=64, num_heads=4):
        super(HierarchicalAttentionNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 音符级别BiLSTM
        self.note_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # 双向LSTM，所以隐藏层大小减半
            bidirectional=True,
            batch_first=True
        )

        # 音符级别注意力
        self.note_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 声部级别BiLSTM
        self.voice_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # 声部级别注意力
        self.voice_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 节拍级别BiLSTM
        self.beat_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # 节拍级别注意力
        self.beat_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 小节级别BiLSTM
        self.measure_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            bidirectional=True,
            batch_first=True
        )

        # 小节级别注意力
        self.measure_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 输出层
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, note_pos, voice_pos, beat_pos, measure_pos):
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, input_dim]
            note_pos: 音符位置 [batch_size, seq_len]
            voice_pos: 声部位置 [batch_size, seq_len]
            beat_pos: 节拍位置 [batch_size, seq_len]
            measure_pos: 小节位置 [batch_size, seq_len]

        Returns:
            output: 模型输出 [batch_size, 1]
        """
        batch_size, seq_len = x.size(0), x.size(1)

        # 1. 音符级别处理
        note_out, _ = self.note_lstm(x)
        note_out, _ = self.note_attention(note_out, note_out, note_out)

        # 2. 声部级别处理
        # 根据声部位置聚合
        voice_out = []
        for i in range(batch_size):
            # 获取每个声部的唯一索引
            unique_voices = torch.unique(voice_pos[i])
            voice_features = []

            for voice_idx in unique_voices:
                if voice_idx < 0:  # 忽略填充值
                    continue
                # 选择该声部的所有音符特征
                mask = (voice_pos[i] == voice_idx)
                if mask.sum() > 0:
                    voice_feat = note_out[i, mask].mean(dim=0, keepdim=True)
                    voice_features.append(voice_feat)

            if voice_features:
                voice_out.append(torch.cat(voice_features, dim=0))
            else:
                # 如果没有有效的声部，使用零填充
                voice_out.append(torch.zeros(1, self.hidden_dim, device=x.device))

        # 处理不同长度的序列
        voice_out = nn.utils.rnn.pad_sequence(voice_out, batch_first=True)

        # 声部BiLSTM和注意力
        voice_out, _ = self.voice_lstm(voice_out)
        voice_out, _ = self.voice_attention(voice_out, voice_out, voice_out)

        # 3. 节拍级别处理（类似声部处理）
        beat_out = []
        for i in range(batch_size):
            unique_beats = torch.unique(beat_pos[i])
            beat_features = []

            for beat_idx in unique_beats:
                if beat_idx < 0:  # 忽略填充值
                    continue
                mask = (beat_pos[i] == beat_idx)
                if mask.sum() > 0:
                    beat_feat = note_out[i, mask].mean(dim=0, keepdim=True)
                    beat_features.append(beat_feat)

            if beat_features:
                beat_out.append(torch.cat(beat_features, dim=0))
            else:
                beat_out.append(torch.zeros(1, self.hidden_dim, device=x.device))

        beat_out = nn.utils.rnn.pad_sequence(beat_out, batch_first=True)

        # 节拍BiLSTM和注意力
        beat_out, _ = self.beat_lstm(beat_out)
        beat_out, _ = self.beat_attention(beat_out, beat_out, beat_out)

        # 4. 小节级别处理（类似节拍处理）
        measure_out = []
        for i in range(batch_size):
            unique_measures = torch.unique(measure_pos[i])
            measure_features = []

            for measure_idx in unique_measures:
                if measure_idx < 0:  # 忽略填充值
                    continue
                mask = (measure_pos[i] == measure_idx)
                if mask.sum() > 0:
                    measure_feat = note_out[i, mask].mean(dim=0, keepdim=True)
                    measure_features.append(measure_feat)

            if measure_features:
                measure_out.append(torch.cat(measure_features, dim=0))
            else:
                measure_out.append(torch.zeros(1, self.hidden_dim, device=x.device))

        measure_out = nn.utils.rnn.pad_sequence(measure_out, batch_first=True)

        # 小节BiLSTM和注意力
        measure_out, _ = self.measure_lstm(measure_out)
        measure_out, _ = self.measure_attention(measure_out, measure_out, measure_out)

        # 5. 使用小节级别的最后输出作为整体特征表示
        final_out = measure_out[:, -1]

        # 输出层
        output = self.fc(final_out)

        return output


# 模型使用示例
if __name__ == "__main__":
    # 创建音高轮廓CNN模型
    pc_model = PitchContourCNN()
    print("音高轮廓CNN模型创建成功")

    # 创建梅尔频谱CRNN模型
    mel_model = MelSpectrogramCRNN()
    print("梅尔频谱CRNN模型创建成功")

    # 创建混合模型
    hybrid_model = HybridPianoModel()
    print("混合模型创建成功")

    # 创建多任务模型
    multi_task_model = MultiTaskPianoModel(hybrid_model)
    print("多任务模型创建成功")

    # 创建分层注意力网络
    han_model = HierarchicalAttentionNetwork(input_dim=128)
    print("分层注意力网络创建成功")

    # 测试前向传播
    batch_size = 2
    seq_len = 1000

    # 音高轮廓输入
    pc_input = torch.randn(batch_size, 1, seq_len)
    pc_output = pc_model(pc_input)
    print(f"音高轮廓CNN输出形状: {pc_output.shape}")

    # 梅尔频谱输入
    mel_input = torch.randn(batch_size, 1, 128, 100)  # [batch, channel, freq, time]
    mel_output = mel_model(mel_input)
    print(f"梅尔频谱CRNN输出形状: {mel_output.shape}")

    # 混合模型输入
    hybrid_output = hybrid_model(pc_input, mel_input)
    print(f"混合模型输出形状: {hybrid_output.shape}")

    # 多任务模型输入
    multi_task_output = multi_task_model(pc_input, mel_input)
    print(f"多任务模型输出形状: {multi_task_output.shape}")

    # 分层注意力网络输入
    x = torch.randn(batch_size, seq_len, 128)
    note_pos = torch.randint(-1, 50, (batch_size, seq_len))
    voice_pos = torch.randint(-1, 4, (batch_size, seq_len))
    beat_pos = torch.randint(-1, 16, (batch_size, seq_len))
    measure_pos = torch.randint(-1, 8, (batch_size, seq_len))

    han_output = han_model(x, note_pos, voice_pos, beat_pos, measure_pos)
    print(f"分层注意力网络输出形状: {han_output.shape}")