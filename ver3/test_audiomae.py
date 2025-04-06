"""
测试AudioMAE模型
"""

import os
import sys
import torch
import logging
import yaml
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio.audiomae import create_audiomae_model
from data.preprocessing.audio_processor import AudioProcessor
from data.datasets.neuropiano_hf import NeuroPianoHFDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audiomae_creation():
    """测试AudioMAE模型创建"""
    try:
        print("Creating AudioMAE model...")
        
        # 确保配置目录存在
        os.makedirs("config", exist_ok=True)
        
        # 创建模型
        model = create_audiomae_model()
        
        # 打印模型信息
        print(f"AudioMAE model created successfully")
        print(f"Model type: {type(model).__name__}")
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        
        # 检查模型是否具有forward方法
        assert hasattr(model, 'forward'), "Model does not have forward method"
        
        return True
    except Exception as e:
        logger.error(f"Error creating AudioMAE model: {e}")
        print(f"Error details: {e}")
        return False

def test_audiomae_inference():
    """测试AudioMAE模型推理"""
    try:
        print("\nTesting AudioMAE inference...")
        
        # 创建模型
        model = create_audiomae_model()
        model.eval()  # 设置为评估模式
        
        # 打印模型参数
        print(f"AudioMAE img_size: {model.img_size}")
        print(f"AudioMAE patch_size: {model.patch_size}")
        
        # 测试不同输入大小
        test_sizes = [
            (2, 1, 128, 4096),  # 原始大小
            (2, 1, 128, 1024),  # 较短的时间步长
        ]
        
        for batch_size, channels, n_mels, time_steps in test_sizes:
            print(f"\nTesting with input shape: {batch_size}x{channels}x{n_mels}x{time_steps}")
            
            # 创建随机输入
            dummy_input = torch.randn(batch_size, channels, n_mels, time_steps)
            print(f"Input shape: {dummy_input.shape}")
            
            # 调整位置嵌入
            if hasattr(model, 'pos_embed'):
                print(f"Original pos_embed shape: {model.pos_embed.shape}")
                
                # 计算patch数量
                h_patches = n_mels // model.patch_size[0]
                w_patches = time_steps // model.patch_size[1]
                n_patches = h_patches * w_patches
                print(f"Expected patches: {n_patches} (= {h_patches} * {w_patches})")
                
                # 创建新的位置嵌入
                with torch.no_grad():
                    # 保存原始位置嵌入
                    original_pos_embed = model.pos_embed.clone()
                    
                    # 创建新的位置嵌入
                    embed_dim = model.pos_embed.shape[2]
                    new_pos_embed = torch.zeros(1, n_patches, embed_dim)  # 移除+1，因为CLS token已经包含在n_patches中
                    
                    # 复制CLS token位置嵌入
                    new_pos_embed[:, 0] = original_pos_embed[:, 0]
                    
                    # 初始化其余部分为随机值
                    new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
                    
                    # 应用新的位置嵌入
                    model.pos_embed = torch.nn.Parameter(new_pos_embed)
                    print(f"New pos_embed shape: {model.pos_embed.shape}")
            
            # 前向传播
            with torch.no_grad():
                output, patch_features = model(dummy_input, return_patch_features=True)
            
            # 打印输出信息
            print(f"Output shape: {output.shape}")
            print(f"Patch features shape: {patch_features.shape}")
            
            # 恢复原始位置嵌入
            if hasattr(model, 'pos_embed'):
                model.pos_embed = torch.nn.Parameter(original_pos_embed)
        
        return True
    except Exception as e:
        logger.error(f"Error testing AudioMAE inference: {e}")
        print(f"Error details: {e}")
        return False

def test_audiomae_with_real_data():
    """使用真实数据测试AudioMAE模型"""
    try:
        print("\nTesting AudioMAE with real data...")
        
        # 创建数据集实例（仅加载少量数据）
        dataset = NeuroPianoHFDataset(
            split="train",
            max_qa_pairs=2  # 仅用于测试，限制加载的样本数
        )
        
        if len(dataset) == 0:
            print("Dataset is empty, skipping test with real data")
            return False
        
        # 创建模型
        model = create_audiomae_model()
        model.eval()  # 设置为评估模式
        
        # 获取第一个样本
        sample = dataset[0]
        mel_spectrogram = sample['mel_spectrogram']
        
        # 添加批次维度（如果需要）
        if len(mel_spectrogram.shape) == 3:  # [C, H, W]
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # [1, C, H, W]
        
        print(f"Input mel spectrogram shape: {mel_spectrogram.shape}")
        
        # 前向传播
        with torch.no_grad():
            output, patch_features = model(mel_spectrogram, return_patch_features=True)
        
        # 打印输出信息
        print(f"Output shape: {output.shape}")
        print(f"Patch features shape: {patch_features.shape}")
        
        # 简单可视化（可选）
        visualize_features = False
        if visualize_features:
            # 将patch features转换为可视化格式
            # 这里我们只显示第一个特征的均值
            mean_features = patch_features[0].mean(dim=1).reshape(-1).cpu().numpy()
            plt.figure(figsize=(10, 5))
            plt.plot(mean_features)
            plt.title('Mean Feature Values')
            plt.savefig('audiomae_features.png')
            plt.close()
            print("Feature visualization saved to audiomae_features.png")
        
        return True
    except Exception as e:
        logger.error(f"Error testing AudioMAE with real data: {e}")
        print(f"Error details: {e}")
        return False

def test_audiomae_with_audio_processor():
    """测试AudioMAE与音频处理器的集成"""
    try:
        print("\nTesting AudioMAE with AudioProcessor...")
        
        # 创建音频处理器
        audio_processor = AudioProcessor()
        
        # 创建模型
        model = create_audiomae_model()
        model.eval()
        
        # 使用随机音频数据
        # 生成一个随机波形作为测试
        sample_rate = audio_processor.sample_rate
        duration = 5  # 5秒
        num_samples = sample_rate * duration
        dummy_waveform = torch.sin(torch.linspace(0, 1000 * np.pi, num_samples)).unsqueeze(0)  # 添加通道维度
        
        # 提取特征
        mel_spec = audio_processor.extract_mel_spectrogram(dummy_waveform)
        normalized_spec = audio_processor.normalize_spectrogram(mel_spec)
        padded_spec = audio_processor.pad_or_truncate(normalized_spec)
        
        # 添加批次维度
        input_spec = padded_spec.unsqueeze(0)  # [1, 1, n_mels, time]
        
        print(f"Processed audio shape: {input_spec.shape}")
        
        # 前向传播
        with torch.no_grad():
            output, patch_features = model(input_spec, return_patch_features=True)
        
        # 打印输出信息
        print(f"Output shape: {output.shape}")
        print(f"Patch features shape: {patch_features.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing AudioMAE with AudioProcessor: {e}")
        print(f"Error details: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING AUDIOMAE MODEL")
    print("=" * 50)
    
    # 测试模型创建
    creation_success = test_audiomae_creation()
    
    # 测试模型推理
    inference_success = test_audiomae_inference()
    
    # 测试与音频处理器集成
    processor_success = test_audiomae_with_audio_processor()
    
    # 测试使用真实数据
    real_data_success = test_audiomae_with_real_data()
    
    # 总结
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"AudioMAE creation test: {'PASSED' if creation_success else 'FAILED'}")
    print(f"AudioMAE inference test: {'PASSED' if inference_success else 'FAILED'}")
    print(f"AudioMAE with AudioProcessor test: {'PASSED' if processor_success else 'FAILED'}")
    print(f"AudioMAE with real data test: {'PASSED' if real_data_success else 'FAILED'}")
    print("=" * 50) 