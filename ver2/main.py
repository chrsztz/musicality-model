#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
钢琴演奏评估系统 - 主程序

该脚本展示了完整的钢琴演奏评估系统工作流，包括：
1. 数据加载和处理
2. 特征提取
3. 模型训练
4. 实时评估

使用方法:
python main.py --mode [train|evaluate|realtime] --model_type [pc_cnn|mel_crnn|hybrid]
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# 导入自定义模块
from dataset_prep import MIDIDataProcessor
from feature_extraction import PianoFeatureExtractor
from model import PitchContourCNN, MelSpectrogramCRNN, HybridPianoModel, MultiTaskPianoModel
from model_training import PianoPerformanceDataset, PitchContourCollate, MelSpectrogramCollate, HybridCollate, \
    ModelTrainer
from realtime_process import RealTimePianoAssessment


# 配置参数
class Config:
    # 数据路径
    data_dir = "./data/all_2rounds"  # MIDI文件目录
    labels_path = "./data/labels/total_2rounds.csv"  # 标签文件路径
    model_save_dir = "./models"  # 模型保存目录

    # 模型训练参数
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    early_stopping_patience = 10

    # 实时评估参数
    sample_rate = 44100
    block_duration = 2.0
    overlap = 0.5

    # 评估维度
    task_names = ['musicality', 'note_accuracy', 'rhythm_accuracy', 'tone_quality']

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 其他参数
    random_seed = 42


def seed_everything(seed):
    """设置随机种子以确保可重复性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_type, multi_task=True, num_tasks=4):
    """
    创建指定类型的模型

    Args:
        model_type: 模型类型 ('pc_cnn', 'mel_crnn', 'hybrid')
        multi_task: 是否为多任务模型
        num_tasks: 任务数量

    Returns:
        model: 创建的模型
    """
    print(f"创建{model_type}模型...")

    if model_type == 'pc_cnn':
        base_model = PitchContourCNN()
    elif model_type == 'mel_crnn':
        base_model = MelSpectrogramCRNN()
    elif model_type == 'hybrid':
        base_model = HybridPianoModel()
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if multi_task:
        return MultiTaskPianoModel(base_model, num_tasks=num_tasks)
    else:
        return base_model


def data_preparation(config):
    """
    准备训练和评估数据

    Args:
        config: 配置对象

    Returns:
        训练、验证和测试数据加载器
    """
    print("加载和处理数据...")

    # 创建数据处理器
    processor = MIDIDataProcessor(config.data_dir, config.labels_path)

    # 处理数据集
    features, labels, valid_files = processor.process_dataset()

    if not features or len(features) == 0:
        raise ValueError("没有加载到有效数据")

    print(f"成功加载 {len(features)} 个有效MIDI文件")

    # 特征提取
    print("提取特征...")
    feature_extractor = PianoFeatureExtractor()

    extracted_features = []
    for midi_feature in tqdm(features, desc="提取特征"):
        # 获取MIDI数据
        midi_path = midi_feature.get('midi_path', '')
        if midi_path and os.path.exists(midi_path):
            midi_data = processor.load_midi(midi_path)
            # 提取所有特征
            extracted_feature = feature_extractor.extract_all_features(midi_data)
            extracted_features.append(extracted_feature)
        else:
            # 使用原始特征
            extracted_features.append(midi_feature)

    # 数据集划分
    print("划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        extracted_features, labels, test_size=0.2, random_state=config.random_seed
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=config.random_seed
    )

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 创建数据集
    train_dataset = PianoPerformanceDataset(X_train, y_train)
    val_dataset = PianoPerformanceDataset(X_val, y_val)
    test_dataset = PianoPerformanceDataset(X_test, y_test)

    # 创建数据整理函数
    pc_collate_fn = PitchContourCollate()
    mel_collate_fn = MelSpectrogramCollate()
    hybrid_collate_fn = HybridCollate()

    # 创建数据加载器
    train_loader_pc = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=pc_collate_fn)
    val_loader_pc = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=pc_collate_fn)
    test_loader_pc = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=pc_collate_fn)

    train_loader_mel = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=mel_collate_fn)
    val_loader_mel = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=mel_collate_fn)
    test_loader_mel = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=mel_collate_fn)

    train_loader_hybrid = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                     collate_fn=hybrid_collate_fn)
    val_loader_hybrid = DataLoader(val_dataset, batch_size=config.batch_size, collate_fn=hybrid_collate_fn)
    test_loader_hybrid = DataLoader(test_dataset, batch_size=config.batch_size, collate_fn=hybrid_collate_fn)

    return {
        'pc_cnn': (train_loader_pc, val_loader_pc, test_loader_pc),
        'mel_crnn': (train_loader_mel, val_loader_mel, test_loader_mel),
        'hybrid': (train_loader_hybrid, val_loader_hybrid, test_loader_hybrid)
    }


def train_model(model, model_type, data_loaders, config):
    """
    训练模型

    Args:
        model: 要训练的模型
        model_type: 模型类型
        data_loaders: 数据加载器字典
        config: 配置对象

    Returns:
        训练好的模型和训练历史
    """
    print(f"开始训练{model_type}模型...")

    # 获取对应的数据加载器
    train_loader, val_loader, _ = data_loaders[model_type]

    # 创建模型训练器
    trainer = ModelTrainer(
        model=model,
        device=config.device,
        model_type=model_type,
        multi_task=(len(config.task_names) > 1)
    )

    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 设置学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 设置训练器的优化器和调度器
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler

    # 创建模型保存路径
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)

    model_save_path = os.path.join(config.model_save_dir, f"{model_type}_model.pth")

    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience,
        save_path=model_save_path
    )

    # 可视化训练历史
    trainer.visualize_training_history(history)

    print(f"模型训练完成，已保存到 {model_save_path}")

    return model, history


def evaluate_model(model, model_type, data_loaders, config):
    """
    评估模型性能

    Args:
        model: 要评估的模型
        model_type: 模型类型
        data_loaders: 数据加载器字典
        config: 配置对象

    Returns:
        评估结果
    """
    print(f"评估{model_type}模型性能...")

    # 获取测试数据加载器
    _, _, test_loader = data_loaders[model_type]

    # 创建模型训练器（用于评估）
    trainer = ModelTrainer(
        model=model,
        device=config.device,
        model_type=model_type,
        multi_task=(len(config.task_names) > 1)
    )

    # 评估模型
    results = trainer.evaluate(test_loader, task_names=config.task_names)

    return results


def start_realtime_assessment(model, model_type, config):
    """
    启动实时评估

    Args:
        model: 评估使用的模型
        model_type: 模型类型
        config: 配置对象
    """
    print(f"启动实时评估，使用{model_type}模型...")

    # 创建实时评估系统
    assessment_system = RealTimePianoAssessment(
        model=model,
        model_type=model_type,
        device=config.device,
        sample_rate=config.sample_rate,
        block_duration=config.block_duration,
        overlap=config.overlap,
        task_names=config.task_names,
        multi_task=(len(config.task_names) > 1)
    )

    # 启动交互式可视化
    assessment_system.start_interactive_visualization()

    # 启动评估
    assessment_system.start()

    # 等待用户停止
    try:
        print("实时评估已启动，开始演奏...")
        print("按Ctrl+C停止录制")
        while True:
            pass
    except KeyboardInterrupt:
        pass

    # 停止评估
    assessment_system.stop()

    # 生成反馈
    feedback = assessment_system.generate_feedback()
    print("\n生成的反馈:")
    print(feedback)

    # 保存MIDI和音频
    assessment_system.save_midi("output/detected_notes.mid")
    assessment_system.save_audio("output/recorded_audio.wav")

    # 显示可视化结果
    assessment_system.visualize()


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="钢琴演奏评估系统")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'realtime'],
                        help='运行模式: train(训练), evaluate(评估), realtime(实时评估)')
    parser.add_argument('--model_type', type=str, default='hybrid', choices=['pc_cnn', 'mel_crnn', 'hybrid'],
                        help='模型类型: pc_cnn(音高轮廓CNN), mel_crnn(梅尔频谱CRNN), hybrid(混合模型)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='预训练模型路径，用于评估或实时评估模式')

    args = parser.parse_args()

    # 配置
    config = Config()

    # 设置随机种子
    seed_everything(config.random_seed)

    print(f"使用设备: {config.device}")

    # 创建输出目录
    if not os.path.exists('output'):
        os.makedirs('output')

    # 创建多任务模型
    model = create_model(args.model_type, multi_task=(len(config.task_names) > 1), num_tasks=len(config.task_names))
    model.to(config.device)

    # 如果提供了模型路径，则加载预训练权重
    if args.model_path:
        if os.path.exists(args.model_path):
            print(f"加载预训练模型: {args.model_path}")
            model.load_state_dict(torch.load(args.model_path, map_location=config.device))
        else:
            print(f"警告: 模型路径 {args.model_path} 不存在")
            if args.mode == 'evaluate' or args.mode == 'realtime':
                print("评估或实时评估需要预训练模型")
                return

    # 根据模式执行不同操作
    if args.mode == 'train':
        # 准备数据
        data_loaders = data_preparation(config)

        # 训练模型
        trained_model, history = train_model(model, args.model_type, data_loaders, config)

        # 评估模型
        results = evaluate_model(trained_model, args.model_type, data_loaders, config)

        print("训练和评估完成!")

    elif args.mode == 'evaluate':
        # 准备数据
        data_loaders = data_preparation(config)

        # 评估模型
        results = evaluate_model(model, args.model_type, data_loaders, config)

        print("评估完成!")

    elif args.mode == 'realtime':
        # 启动实时评估
        start_realtime_assessment(model, args.model_type, config)

        print("实时评估结束!")


if __name__ == "__main__":
    main()