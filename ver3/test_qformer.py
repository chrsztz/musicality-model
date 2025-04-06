"""
测试Q-former模型
"""

import os
import sys
import torch
import logging
import yaml
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion.qformer import create_audio_qformer
from models.audio.audiomae import create_audiomae_model
from data.preprocessing.audio_processor import AudioProcessor
from data.datasets.neuropiano_hf import NeuroPianoHFDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_qformer_creation():
    """测试Q-former模型创建"""
    try:
        print("Creating Q-former model...")
        
        # 确保配置目录存在
        os.makedirs("config", exist_ok=True)
        
        # 创建模型
        qformer = create_audio_qformer()
        
        # 打印模型信息
        print(f"Q-former model created successfully")
        print(f"Model type: {type(qformer).__name__}")
        print(f"Number of parameters: {sum(p.numel() for p in qformer.parameters())}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating Q-former model: {e}")
        print(f"Error details: {e}")
        return False

def test_qformer_inference():
    """测试Q-former模型推理"""
    try:
        print("\nTesting Q-former inference...")
        
        # 创建AudioMAE模型作为输入源
        audiomae = create_audiomae_model()
        audiomae.eval()
        
        # 创建Q-former模型
        qformer = create_audio_qformer()
        qformer.eval()
        
        # 创建随机输入
        batch_size = 2
        n_mels = 128
        time_steps = 4096
        
        # 创建随机音频输入
        dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
        print(f"Original input shape: {dummy_input.shape}")
        
        # 通过AudioMAE获取特征
        with torch.no_grad():
            audio_features, _ = audiomae(dummy_input, return_patch_features=True)
        
        print(f"AudioMAE output shape: {audio_features.shape}")
        
        # 通过Q-former获取查询嵌入
        with torch.no_grad():
            query_embeddings = qformer(audio_features)
        
        print(f"Q-former output shape: {query_embeddings.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Q-former inference: {e}")
        print(f"Error details: {e}")
        return False

def test_qformer_with_real_data():
    """使用真实数据测试Q-former"""
    try:
        print("\nTesting Q-former with real data...")
        
        # 创建数据集实例（仅加载少量数据）
        dataset = NeuroPianoHFDataset(
            split="train",
            max_qa_pairs=1  # 仅用于测试
        )
        
        if len(dataset) == 0:
            print("Dataset is empty, skipping test with real data")
            return False
        
        # 创建AudioMAE模型
        audiomae = create_audiomae_model()
        audiomae.eval()
        
        # 创建Q-former模型
        qformer = create_audio_qformer()
        qformer.eval()
        
        # 获取第一个样本
        sample = dataset[0]
        mel_spectrogram = sample['mel_spectrogram']
        question = sample['question']
        print(f"Question: {question}")
        
        # 添加批次维度（如果需要）
        if len(mel_spectrogram.shape) == 3:  # [C, H, W]
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # [1, C, H, W]
        
        print(f"Input mel spectrogram shape: {mel_spectrogram.shape}")
        
        # 通过AudioMAE获取特征
        with torch.no_grad():
            audio_features, _ = audiomae(mel_spectrogram, return_patch_features=True)
        
        print(f"AudioMAE output shape: {audio_features.shape}")
        
        # 通过Q-former获取查询嵌入
        with torch.no_grad():
            query_embeddings = qformer(audio_features)
        
        print(f"Q-former output shape: {query_embeddings.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing Q-former with real data: {e}")
        print(f"Error details: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING Q-FORMER MODEL")
    print("=" * 50)
    
    # 测试模型创建
    creation_success = test_qformer_creation()
    
    # 测试模型推理
    inference_success = test_qformer_inference()
    
    # 测试使用真实数据
    real_data_success = test_qformer_with_real_data()
    
    # 总结
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Q-former creation test: {'PASSED' if creation_success else 'FAILED'}")
    print(f"Q-former inference test: {'PASSED' if inference_success else 'FAILED'}")
    print(f"Q-former with real data test: {'PASSED' if real_data_success else 'FAILED'}")
    print("=" * 50) 