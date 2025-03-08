import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


class PianoPerformanceDataset(Dataset):
    """钢琴演奏数据集类"""

    def __init__(self, features, labels=None, transform=None):
        """
        初始化数据集

        Args:
            features: 特征列表，每个元素是一个特征字典
            labels: 标签列表，每个元素是一个标签字典
            transform: 特征转换函数
        """
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """获取单个样本"""
        feature = self.features[idx]

        # 应用转换（如果有）
        if self.transform:
            feature = self.transform(feature)

        # 如果没有标签，则只返回特征
        if self.labels is None:
            return feature

        # 获取标签
        label = self.labels[idx]

        return feature, label


class PitchContourCollate:
    """音高轮廓数据整理函数，处理不同长度的序列"""

    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        """
        整理一个批次的数据

        Args:
            batch: 一个批次的样本列表，每个样本为(feature_dict, label_dict)

        Returns:
            整理后的批次数据，包括填充后的特征和标签
        """
        # 分离特征和标签
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # 有标签的情况
            features = [item[0] for item in batch]
            labels = [item[1] for item in batch]
        else:
            # 无标签的情况
            features = batch
            labels = None

        # 提取音高轮廓
        pitch_contours = [feat['pitch_contour'] for feat in features]

        # 找出最大长度
        max_len = max(len(pc) for pc in pitch_contours)

        # 填充序列
        padded_pcs = []
        for pc in pitch_contours:
            # 创建填充序列
            padded = np.full(max_len, self.pad_value, dtype=np.float32)
            # 复制原始数据
            padded[:len(pc)] = pc
            padded_pcs.append(padded)

        # 转换为torch张量
        padded_pcs = torch.tensor(padded_pcs, dtype=torch.float32).unsqueeze(1)  # [batch, 1, seq_len]

        # 提取其他特征
        other_features = []
        for feat in features:
            # 复制特征字典并移除音高轮廓
            feat_copy = feat.copy()
            if 'pitch_contour' in feat_copy:
                del feat_copy['pitch_contour']

            # 扁平化特征
            flat_features = []
            for key, value in feat_copy.items():
                if isinstance(value, (list, np.ndarray)):
                    flat_features.extend(value)
                else:
                    flat_features.append(value)

            other_features.append(flat_features)

        # 转换为torch张量
        if other_features:
            other_features = torch.tensor(other_features, dtype=torch.float32)
        else:
            other_features = None

        # 处理标签
        if labels is not None:
            # 提取标签值，转换为张量
            if isinstance(labels[0], dict):
                # 字典标签 - 提取所有标签键并创建张量
                label_keys = list(labels[0].keys())
                label_tensors = {}

                for key in label_keys:
                    if key in ['musicality', 'note_accuracy', 'rhythm_accuracy', 'tone_quality']:
                        values = [label[key] for label in labels]
                        label_tensors[key] = torch.tensor(values, dtype=torch.float32)

                # 如果需要单个张量，则可以合并所有目标标签
                target_keys = ['musicality', 'note_accuracy', 'rhythm_accuracy', 'tone_quality']
                target_values = []
                for label in labels:
                    values = [label.get(key, 0) for key in target_keys if key in label]
                    target_values.append(values)

                if target_values:
                    target_tensor = torch.tensor(target_values, dtype=torch.float32)
                else:
                    target_tensor = None
            else:
                # 数值标签
                target_tensor = torch.tensor(labels, dtype=torch.float32)

            return {'pitch_contour': padded_pcs, 'other_features': other_features}, target_tensor
        else:
            return {'pitch_contour': padded_pcs, 'other_features': other_features}


class MelSpectrogramCollate:
    """梅尔频谱图数据整理函数，处理不同长度的频谱图"""

    def __init__(self, pad_value=0):
        self.pad_value = pad_value

    def __call__(self, batch):
        """
        整理一个批次的数据

        Args:
            batch: 一个批次的样本列表，每个样本为(feature_dict, label_dict)

        Returns:
            整理后的批次数据，包括填充后的特征和标签
        """
        # 分离特征和标签
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # 有标签的情况
            features = [item[0] for item in batch]
            labels = [item[1] for item in batch]
        else:
            # 无标签的情况
            features = batch
            labels = None

        # 提取梅尔频谱图
        mel_specs = [feat['mel_spectrogram'] for feat in features if 'mel_spectrogram' in feat]

        # 如果没有梅尔频谱图，返回None
        if not mel_specs:
            return None, None if labels is not None else None

        # 找出最大频率维度和时间维度
        max_freq = max(spec.shape[0] for spec in mel_specs)
        max_time = max(spec.shape[1] for spec in mel_specs)

        # 填充频谱图
        padded_specs = []
        for spec in mel_specs:
            freq_dim, time_dim = spec.shape
            # 创建填充频谱图
            padded = np.full((max_freq, max_time), self.pad_value, dtype=np.float32)
            # 复制原始数据
            padded[:freq_dim, :time_dim] = spec
            padded_specs.append(padded)

        # 转换为torch张量
        padded_specs = torch.tensor(padded_specs, dtype=torch.float32).unsqueeze(1)  # [batch, 1, freq, time]

        # 提取其他特征（与PitchContourCollate相同）
        other_features = []
        for feat in features:
            # 复制特征字典并移除梅尔频谱图
            feat_copy = feat.copy()
            if 'mel_spectrogram' in feat_copy:
                del feat_copy['mel_spectrogram']

            # 扁平化特征
            flat_features = []
            for key, value in feat_copy.items():
                if isinstance(value, (list, np.ndarray)) and key != 'pitch_contour':
                    flat_features.extend(value)
                elif key != 'pitch_contour':
                    flat_features.append(value)

            other_features.append(flat_features)

        # 转换为torch张量
        if other_features:
            other_features = torch.tensor(other_features, dtype=torch.float32)
        else:
            other_features = None

        # 处理标签（与PitchContourCollate相同）
        if labels is not None:
            # 提取标签值，转换为张量
            if isinstance(labels[0], dict):
                # 字典标签 - 提取所有标签键并创建张量
                label_keys = list(labels[0].keys())
                label_tensors = {}

                for key in label_keys:
                    if key in ['musicality', 'note_accuracy', 'rhythm_accuracy', 'tone_quality']:
                        values = [label[key] for label in labels]
                        label_tensors[key] = torch.tensor(values, dtype=torch.float32)

                # 如果需要单个张量，则可以合并所有目标标签
                target_keys = ['musicality', 'note_accuracy', 'rhythm_accuracy', 'tone_quality']
                target_values = []
                for label in labels:
                    values = [label.get(key, 0) for key in target_keys if key in label]
                    target_values.append(values)

                if target_values:
                    target_tensor = torch.tensor(target_values, dtype=torch.float32)
                else:
                    target_tensor = None
            else:
                # 数值标签
                target_tensor = torch.tensor(labels, dtype=torch.float32)

            return {'mel_spectrogram': padded_specs, 'other_features': other_features}, target_tensor
        else:
            return {'mel_spectrogram': padded_specs, 'other_features': other_features}


class HybridCollate:
    """混合数据整理函数，处理音高轮廓和梅尔频谱图"""

    def __init__(self, pad_value=0):
        self.pad_value = pad_value
        self.pc_collate = PitchContourCollate(pad_value)
        self.mel_collate = MelSpectrogramCollate(pad_value)

    def __call__(self, batch):
        """
        整理一个批次的数据

        Args:
            batch: 一个批次的样本列表，每个样本为(feature_dict, label_dict)

        Returns:
            整理后的批次数据，包括填充后的特征和标签
        """
        # 分离特征和标签
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            # 有标签的情况
            features = [item[0] for item in batch]
            labels = [item[1] for item in batch]
        else:
            # 无标签的情况
            features = batch
            labels = None

        # 使用各自的collate函数处理
        pc_batch = self.pc_collate([(feat, label) if labels is not None else feat
                                    for feat, label in zip(features, labels)] if labels is not None else features)

        mel_batch = self.mel_collate([(feat, label) if labels is not None else feat
                                      for feat, label in zip(features, labels)] if labels is not None else features)

        # 合并结果
        if labels is not None:
            pc_features, _ = pc_batch
            mel_features, target_tensor = mel_batch

            combined_features = {
                'pitch_contour': pc_features['pitch_contour'],
                'mel_spectrogram': mel_features['mel_spectrogram'],
                'other_features': pc_features['other_features']  # 使用pitch_contour的other_features
            }

            return combined_features, target_tensor
        else:
            combined_features = {
                'pitch_contour': pc_batch['pitch_contour'],
                'mel_spectrogram': mel_batch['mel_spectrogram'],
                'other_features': pc_batch['other_features']
            }

            return combined_features


class ModelTrainer:
    """模型训练器类"""

    def __init__(self, model, device=None, criterion=None, optimizer=None, scheduler=None,
                 model_type='pc_cnn', multi_task=False):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            device: 训练设备（CPU或GPU）
            criterion: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            model_type: 模型类型 ('pc_cnn', 'mel_crnn', 'hybrid')
            multi_task: 是否为多任务模型
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = criterion if criterion else nn.MSELoss()
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_type = model_type
        self.multi_task = multi_task

        # 将模型移动到指定设备
        self.model.to(self.device)

    def train(self, train_loader, val_loader, num_epochs=50,
              early_stopping_patience=10, save_path=None):
        """
        训练模型

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_path: 模型保存路径

        Returns:
            训练历史记录
        """
        # 检查优化器
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_r2': [],
            'val_r2': []
        }

        # 早停变量
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Training"):
                # 提取特征和标签
                features, targets = batch

                # 根据模型类型处理输入
                if self.model_type == 'pc_cnn':
                    inputs = features['pitch_contour'].to(self.device)
                elif self.model_type == 'mel_crnn':
                    inputs = features['mel_spectrogram'].to(self.device)
                else:  # hybrid
                    pc_inputs = features['pitch_contour'].to(self.device)
                    mel_inputs = features['mel_spectrogram'].to(self.device)
                    inputs = (pc_inputs, mel_inputs)

                targets = targets.to(self.device)

                # 前向传播
                if self.model_type == 'hybrid':
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)

                # 计算损失
                if self.multi_task:
                    # 多任务损失 - 计算每个任务的损失并求和
                    loss = 0
                    for i in range(outputs.size(1)):
                        loss += self.criterion(outputs[:, i].view(-1, 1), targets[:, i].view(-1, 1))
                else:
                    loss = self.criterion(outputs, targets)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 累计损失
                train_loss += loss.item() * targets.size(0)

                # 保存预测和目标（用于计算R²）
                train_preds.append(outputs.detach().cpu().numpy())
                train_targets.append(targets.cpu().numpy())

            # 计算平均训练损失
            train_loss /= len(train_loader.dataset)

            # 合并预测和目标
            train_preds = np.concatenate(train_preds, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)

            # 计算R²
            train_r2 = self._calculate_r2(train_preds, train_targets)

            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} - Validation"):
                    # 提取特征和标签
                    features, targets = batch

                    # 根据模型类型处理输入
                    if self.model_type == 'pc_cnn':
                        inputs = features['pitch_contour'].to(self.device)
                    elif self.model_type == 'mel_crnn':
                        inputs = features['mel_spectrogram'].to(self.device)
                    else:  # hybrid
                        pc_inputs = features['pitch_contour'].to(self.device)
                        mel_inputs = features['mel_spectrogram'].to(self.device)
                        inputs = (pc_inputs, mel_inputs)

                    targets = targets.to(self.device)

                    # 前向传播
                    if self.model_type == 'hybrid':
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(inputs)

                    # 计算损失
                    if self.multi_task:
                        # 多任务损失
                        loss = 0
                        for i in range(outputs.size(1)):
                            loss += self.criterion(outputs[:, i].view(-1, 1), targets[:, i].view(-1, 1))
                    else:
                        loss = self.criterion(outputs, targets)

                    # 累计损失
                    val_loss += loss.item() * targets.size(0)

                    # 保存预测和目标
                    val_preds.append(outputs.cpu().numpy())
                    val_targets.append(targets.cpu().numpy())

            # 计算平均验证损失
            val_loss /= len(val_loader.dataset)

            # 合并预测和目标
            val_preds = np.concatenate(val_preds, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)

            # 计算R²
            val_r2 = self._calculate_r2(val_preds, val_targets)

            # 更新历史记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_r2'].append(train_r2)
            history['val_r2'].append(val_r2)

            # 打印进度
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, R²: {train_r2:.4f} - "
                  f"Val Loss: {val_loss:.4f}, R²: {val_r2:.4f}")

            # 学习率调度
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()

                # 保存最佳模型
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"早停：验证损失 {early_stopping_patience} 轮未改善")
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history

    def evaluate(self, test_loader, task_names=None):
        """
        评估模型性能

        Args:
            test_loader: 测试数据加载器
            task_names: 多任务模型的任务名称列表

        Returns:
            评估结果字典
        """
        self.model.eval()

        # 存储预测和目标
        all_preds = []
        all_targets = []
        test_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="评估中"):
                # 提取特征和标签
                features, targets = batch

                # 根据模型类型处理输入
                if self.model_type == 'pc_cnn':
                    inputs = features['pitch_contour'].to(self.device)
                elif self.model_type == 'mel_crnn':
                    inputs = features['mel_spectrogram'].to(self.device)
                else:  # hybrid
                    pc_inputs = features['pitch_contour'].to(self.device)
                    mel_inputs = features['mel_spectrogram'].to(self.device)
                    inputs = (pc_inputs, mel_inputs)

                targets = targets.to(self.device)

                # 前向传播
                if self.model_type == 'hybrid':
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)

                # 计算损失
                if self.multi_task:
                    # 多任务损失
                    loss = 0
                    for i in range(outputs.size(1)):
                        loss += self.criterion(outputs[:, i].view(-1, 1), targets[:, i].view(-1, 1))
                else:
                    loss = self.criterion(outputs, targets)

                # 累计损失
                test_loss += loss.item() * targets.size(0)

                # 保存预测和目标
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        # 计算平均测试损失
        test_loss /= len(test_loader.dataset)

        # 合并预测和目标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # 计算整体R²
        r2 = self._calculate_r2(all_preds, all_targets)

        # 计算每个任务的MSE和R²（如果是多任务模型）
        task_metrics = {}
        if self.multi_task:
            task_names = task_names or [f"Task {i + 1}" for i in range(all_preds.shape[1])]

            for i, task in enumerate(task_names):
                task_mse = np.mean((all_preds[:, i] - all_targets[:, i]) ** 2)
                task_r2 = 1 - (np.sum((all_targets[:, i] - all_preds[:, i]) ** 2) /
                               np.sum((all_targets[:, i] - np.mean(all_targets[:, i])) ** 2))

                task_metrics[task] = {
                    'MSE': task_mse,
                    'R²': task_r2
                }

        # 打印评估结果
        print(f"测试损失: {test_loss:.4f}, R²: {r2:.4f}")

        if task_metrics:
            print("\n各任务评估指标:")
            for task, metrics in task_metrics.items():
                print(f"{task}: MSE = {metrics['MSE']:.4f}, R² = {metrics['R²']:.4f}")

        # 可视化预测结果
        if task_names and self.multi_task:
            self._plot_predictions(all_targets, all_preds, task_names)
        else:
            self._plot_predictions(all_targets, all_preds, ['预测'])

        # 返回评估结果
        return {
            'test_loss': test_loss,
            'r2': r2,
            'task_metrics': task_metrics,
            'predictions': all_preds,
            'targets': all_targets
        }

    def predict(self, test_loader, task_names=None):
        """
        使用模型进行预测

        Args:
            test_loader: 测试数据加载器
            task_names: 多任务模型的任务名称列表

        Returns:
            预测结果数组
        """
        self.model.eval()

        # 存储预测
        all_preds = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="预测中"):
                # 提取特征（可能没有标签）
                if isinstance(batch, tuple) and len(batch) == 2:
                    features, _ = batch
                else:
                    features = batch

                # 根据模型类型处理输入
                if self.model_type == 'pc_cnn':
                    inputs = features['pitch_contour'].to(self.device)
                elif self.model_type == 'mel_crnn':
                    inputs = features['mel_spectrogram'].to(self.device)
                else:  # hybrid
                    pc_inputs = features['pitch_contour'].to(self.device)
                    mel_inputs = features['mel_spectrogram'].to(self.device)
                    inputs = (pc_inputs, mel_inputs)

                # 前向传播
                if self.model_type == 'hybrid':
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(inputs)

                # 保存预测
                all_preds.append(outputs.cpu().numpy())

        # 合并预测
        all_preds = np.concatenate(all_preds, axis=0)

        # 如果是多任务模型，为预测结果创建DataFrame
        if task_names and self.multi_task:
            predictions_df = pd.DataFrame(all_preds, columns=task_names)
            return predictions_df
        else:
            return all_preds

    def _calculate_r2(self, predictions, targets):
        """计算R²决定系数"""
        # 如果是多任务模型，计算每个任务的R²并取平均值
        if predictions.shape[1] > 1:
            r2_values = []
            for i in range(predictions.shape[1]):
                ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
                ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
                r2_values.append(1 - (ss_res / ss_tot) if ss_tot > 0 else 0)
            return np.mean(r2_values)
        else:
            # 单任务模型
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _plot_predictions(self, targets, predictions, task_names):
        """绘制预测结果散点图"""
        n_tasks = predictions.shape[1]
        fig, axes = plt.subplots(n_tasks, 1, figsize=(10, 5 * n_tasks))
        axes = [axes] if n_tasks == 1 else axes

        for i, (ax, task) in enumerate(zip(axes, task_names)):
            ax.scatter(targets[:, i], predictions[:, i], alpha=0.5)

            # 添加对角线（理想预测线）
            min_val = min(np.min(targets[:, i]), np.min(predictions[:, i]))
            max_val = max(np.max(targets[:, i]), np.max(predictions[:, i]))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')

            # 计算相关系数
            corr = np.corrcoef(targets[:, i], predictions[:, i])[0, 1]

            ax.set_title(f'{task}: 相关系数 = {corr:.4f}')
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('prediction_results.png')
        plt.close()

    def visualize_training_history(self, history):
        """可视化训练历史"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 绘制损失
        ax1.plot(history['train_loss'], label='训练损失')
        ax1.plot(history['val_loss'], label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)

        # 绘制R²
        ax2.plot(history['train_r2'], label='训练 R²')
        ax2.plot(history['val_r2'], label='验证 R²')
        ax2.set_title('训练和验证 R²')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('R²')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()


# 使用示例
if __name__ == "__main__":
    import random
    from model import PitchContourCNN, MelSpectrogramCRNN, HybridPianoModel, MultiTaskPianoModel

    # 生成模拟数据
    num_samples = 100
    seq_len = 1000
    freq_bins = 128
    time_frames = 100

    # 创建模拟特征
    features = []
    for i in range(num_samples):
        # 生成随机音高轮廓
        pitch_contour = np.random.rand(random.randint(800, 1200))

        # 生成随机梅尔频谱图
        mel_spec = np.random.rand(freq_bins, random.randint(80, 120))

        # 创建特征字典
        feature = {
            'pitch_contour': pitch_contour,
            'mel_spectrogram': mel_spec,
            'beat_stability': random.random(),
            'rhythm_regularity': random.random(),
            'note_length_ratio': random.random(),
            'dynamic_range': random.random()
        }

        features.append(feature)

    # 创建模拟标签
    labels = []
    for i in range(num_samples):
        # 生成四个随机评分
        label = {
            'musicality': random.random(),
            'note_accuracy': random.random(),
            'rhythm_accuracy': random.random(),
            'tone_quality': random.random()
        }

        labels.append(label)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 创建数据集
    train_dataset = PianoPerformanceDataset(X_train, y_train)
    val_dataset = PianoPerformanceDataset(X_val, y_val)
    test_dataset = PianoPerformanceDataset(X_test, y_test)

    # 创建数据加载器
    pc_collate_fn = PitchContourCollate()
    mel_collate_fn = MelSpectrogramCollate()
    hybrid_collate_fn = HybridCollate()

    batch_size = 8

    # 1. 使用音高轮廓CNN模型
    print("使用音高轮廓CNN模型")
    pc_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pc_collate_fn)
    pc_val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=pc_collate_fn)
    pc_test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=pc_collate_fn)

    pc_model = PitchContourCNN()
    pc_trainer = ModelTrainer(pc_model, model_type='pc_cnn')

    # 2. 使用梅尔频谱CRNN模型
    print("使用梅尔频谱CRNN模型")
    mel_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=mel_collate_fn)
    mel_val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=mel_collate_fn)
    mel_test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=mel_collate_fn)

    mel_model = MelSpectrogramCRNN()
    mel_trainer = ModelTrainer(mel_model, model_type='mel_crnn')

    # 3. 使用混合模型
    print("使用混合模型")
    hybrid_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=hybrid_collate_fn)
    hybrid_val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=hybrid_collate_fn)
    hybrid_test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=hybrid_collate_fn)

    hybrid_model = HybridPianoModel()
    hybrid_trainer = ModelTrainer(hybrid_model, model_type='hybrid')

    # 4. 使用多任务模型
    print("使用多任务混合模型")
    multi_task_model = MultiTaskPianoModel(hybrid_model, num_tasks=4)
    multi_task_trainer = ModelTrainer(multi_task_model, model_type='hybrid', multi_task=True)

    # 训练和评估其中一个模型
    print("开始训练混合多任务模型")
    history = multi_task_trainer.train(
        hybrid_train_loader, hybrid_val_loader,
        num_epochs=5, early_stopping_patience=3,
        save_path='models/hybrid_multi_task_model.pth'
    )

    # 可视化训练历史
    multi_task_trainer.visualize_training_history(history)

    # 评估模型
    task_names = ['musicality', 'note_accuracy', 'rhythm_accuracy', 'tone_quality']
    results = multi_task_trainer.evaluate(hybrid_test_loader, task_names=task_names)

    print("评估结果:", results['r2'])