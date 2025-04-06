"""
测试完整模型管线
"""

import os
import sys
import torch
import logging
import yaml
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion.multimodal_fusion import create_piano_performance_model
from data.preprocessing.audio_processor import AudioProcessor
from data.datasets.neuropiano_hf import NeuroPianoHFDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_creation():
    """测试完整模型创建"""
    try:
        print("Creating piano performance model...")
        
        # 确保配置目录存在
        os.makedirs("config", exist_ok=True)
        
        # 创建模型 (使用CPU加速测试)
        model = create_piano_performance_model(device="cpu")
        
        # 打印模型信息
        print(f"Piano performance model created successfully")
        print(f"Model type: {type(model).__name__}")
        
        # 检查模型组件
        print("\nModel components:")
        print(f"- AudioMAE: {type(model.audio_encoder).__name__}")
        print(f"- QFormer: {type(model.qformer).__name__}")
        print(f"- LLM Interface: {type(model.llm_interface).__name__}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating piano performance model: {e}")
        print(f"Error details: {e}")
        return False

def test_model_forward():
    """测试模型前向传播"""
    try:
        print("\nTesting model forward pass...")
        
        # 创建模型 (使用CPU加速测试)
        model = create_piano_performance_model(device="cpu")
        model.eval()
        
        # 创建随机输入
        batch_size = 1
        n_mels = 128
        time_steps = 4096
        
        dummy_input = torch.randn(batch_size, 1, n_mels, time_steps)
        dummy_question = "What's good about this performance?"
        print(f"Input shape: {dummy_input.shape}")
        print(f"Question: {dummy_question}")
        
        # 前向传播
        print("\nExecuting forward pass (this might take some time due to LLM)...")
        try:
            with torch.no_grad():
                # 注意：如果LLM未加载，这可能会失败
                # 通常需要下载和加载预训练的LLM模型
                outputs = model.forward(
                    mel_spectrogram=dummy_input,
                    question=dummy_question,
                    max_new_tokens=20
                )
            
            print(f"Output type: {type(outputs)}")
            print(f"Model generated responses successfully")
            
            if outputs:
                print(f"Sample response: {outputs[0][:100]}...")
                
            full_success = True
        except Exception as e:
            logger.warning(f"Full forward pass failed (this is expected if LLM is not available): {e}")
            print(f"Testing partial forward pass instead...")
            
            # 测试部分前向传播（不包含LLM）
            with torch.no_grad():
                # 1. 通过AudioMAE提取音频特征
                audio_features, _ = model.audio_encoder(dummy_input, return_patch_features=True)
                print(f"AudioMAE output shape: {audio_features.shape}")
                
                # 2. 使用Q-former提取查询特征
                query_embeds = model.qformer(audio_features)
                print(f"QFormer output shape: {query_embeds.shape}")
                
                # LLM接口测试只能检查投影层是否可用
                if hasattr(model.llm_interface, 'llm_proj'):
                    projected_embeds = model.llm_interface.llm_proj(query_embeds)
                    print(f"LLM projection output shape: {projected_embeds.shape}")
                
            full_success = False
            
        return full_success
    except Exception as e:
        logger.error(f"Error testing model forward pass: {e}")
        print(f"Error details: {e}")
        return False

def test_model_with_real_data():
    """使用真实数据测试模型"""
    try:
        print("\nTesting model with real data...")
        
        # 创建数据集实例（仅加载少量数据）
        dataset = NeuroPianoHFDataset(
            split="train",
            max_qa_pairs=1  # 仅用于测试
        )
        
        if len(dataset) == 0:
            print("Dataset is empty, skipping test with real data")
            return False
        
        # 创建模型 (使用CPU加速测试)
        model = create_piano_performance_model(device="cpu")
        model.eval()
        
        # 获取第一个样本
        sample = dataset[0]
        mel_spectrogram = sample['mel_spectrogram']
        question = sample['question']
        answer = sample['answer']
        
        print(f"Question: {question}")
        print(f"Ground truth answer: {answer[:100]}...")
        
        # 添加批次维度（如果需要）
        if len(mel_spectrogram.shape) == 3:  # [C, H, W]
            mel_spectrogram = mel_spectrogram.unsqueeze(0)  # [1, C, H, W]
        
        print(f"Input mel spectrogram shape: {mel_spectrogram.shape}")
        
        # 部分前向传播测试（避免依赖LLM）
        with torch.no_grad():
            # 1. 通过AudioMAE提取音频特征
            audio_features, _ = model.audio_encoder(mel_spectrogram, return_patch_features=True)
            print(f"AudioMAE output shape: {audio_features.shape}")
            
            # 2. 使用Q-former提取查询特征
            query_embeds = model.qformer(audio_features)
            print(f"QFormer output shape: {query_embeds.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing model with real data: {e}")
        print(f"Error details: {e}")
        return False

def test_audio_processor_integration():
    """测试音频处理器与模型的集成"""
    try:
        print("\nTesting audio processor integration...")
        
        # 创建音频处理器
        audio_processor = AudioProcessor()
        
        # 创建模型 (使用CPU加速测试)
        model = create_piano_performance_model(device="cpu")
        model.eval()
        
        # 使用随机音频数据
        # 生成一个随机波形作为测试
        sample_rate = audio_processor.sample_rate
        duration = 5  # 5秒
        num_samples = sample_rate * duration
        dummy_waveform = torch.sin(torch.linspace(0, 1000 * np.pi, num_samples)).unsqueeze(0)  # 添加通道维度
        
        print(f"Generated dummy waveform with {duration} seconds at {sample_rate}Hz")
        
        # 提取特征
        mel_spec = audio_processor.extract_mel_spectrogram(dummy_waveform)
        normalized_spec = audio_processor.normalize_spectrogram(mel_spec)
        padded_spec = audio_processor.pad_or_truncate(normalized_spec)
        
        # 添加批次维度
        input_spec = padded_spec.unsqueeze(0)  # [1, 1, n_mels, time]
        
        print(f"Processed audio shape: {input_spec.shape}")
        
        # 部分前向传播测试
        with torch.no_grad():
            # 1. 通过AudioMAE提取音频特征
            audio_features, _ = model.audio_encoder(input_spec, return_patch_features=True)
            print(f"AudioMAE output shape: {audio_features.shape}")
            
            # 2. 使用Q-former提取查询特征
            query_embeds = model.qformer(audio_features)
            print(f"QFormer output shape: {query_embeds.shape}")
        
        return True
    except Exception as e:
        logger.error(f"Error testing audio processor integration: {e}")
        print(f"Error details: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING COMPLETE MODEL PIPELINE")
    print("=" * 50)
    
    # 测试模型创建
    creation_success = test_model_creation()
    
    # 测试模型前向传播
    forward_success = test_model_forward()
    
    # 测试音频处理器集成
    processor_success = test_audio_processor_integration()
    
    # 测试使用真实数据
    real_data_success = test_model_with_real_data()
    
    # 总结
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Model creation test: {'PASSED' if creation_success else 'FAILED'}")
    print(f"Model forward pass test: {'PASSED' if forward_success else 'PARTIAL (LLM not tested)'}")
    print(f"Audio processor integration test: {'PASSED' if processor_success else 'FAILED'}")
    print(f"Model with real data test: {'PASSED' if real_data_success else 'FAILED'}")
    print("=" * 50) 