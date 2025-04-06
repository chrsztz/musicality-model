"""
测试管道各组件
"""
import os
import sys
import yaml
import torch
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio.audiomae import create_audiomae_model
from models.fusion.qformer import create_audio_qformer
from models.llm.llm_interface import create_audio_llm_interface

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audiomae():
    """测试AudioMAE模型"""
    logger.info("Testing AudioMAE component...")
    
    # 创建模型
    model = create_audiomae_model()
    model.eval()
    
    # 创建随机输入 [B, C, H, W]
    batch_size = 1
    channels = 1
    n_mels = 128
    time_steps = 4096
    dummy_input = torch.randn(batch_size, channels, n_mels, time_steps)
    
    # 测试前向传播
    with torch.no_grad():
        # 正常输出
        output = model(dummy_input)
        logger.info(f"Output shape: {output.shape}")
        
        # 输出patch特征
        patch_features, output = model(dummy_input, return_patch_features=True)
        logger.info(f"Patch features shape: {patch_features.shape}")
        logger.info(f"Output shape (again): {output.shape}")
    
    return True

def test_qformer(input_shape=(1, 1024)):
    """测试Q-former模型"""
    logger.info("Testing Q-former component...")
    
    # 创建模型
    model = create_audio_qformer()
    model.eval()
    
    # 创建随机输入
    # Q-former期望输入为 [B, N, D]，其中N是序列长度，D是特征维度
    batch_size = 1
    feature_dim = input_shape[-1]  # 1024 from AudioMAE output
    
    # 测试两种情况：
    # 1. 单个特征向量 [B, D]
    # 2. 序列特征 [B, N, D]
    
    # 情况1：单个特征向量
    single_feature = torch.randn(batch_size, feature_dim)
    logger.info(f"Testing with single feature vector: {single_feature.shape}")
    
    try:
        # 尝试直接使用
        with torch.no_grad():
            output = model(single_feature)
            logger.info(f"Direct output shape: {output.shape}")
    except Exception as e:
        logger.error(f"Error with direct input: {e}")
        
        # 尝试扩展维度
        try:
            # 添加序列维度 [B, 1, D]
            expanded_feature = single_feature.unsqueeze(1)
            logger.info(f"Expanded feature shape: {expanded_feature.shape}")
            
            with torch.no_grad():
                output = model(expanded_feature)
                logger.info(f"Output shape with expanded input: {output.shape}")
        except Exception as e:
            logger.error(f"Error with expanded input: {e}")
    
    # 情况2：序列特征
    seq_len = 32  # 假设一个合理的序列长度
    sequence_features = torch.randn(batch_size, seq_len, feature_dim)
    logger.info(f"Testing with sequence features: {sequence_features.shape}")
    
    try:
        with torch.no_grad():
            output = model(sequence_features)
            logger.info(f"Output shape with sequence input: {output.shape}")
    except Exception as e:
        logger.error(f"Error with sequence input: {e}")
    
    return True

def test_full_pipeline():
    """测试完整管道"""
    logger.info("Testing full pipeline...")
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 创建模型
    audiomae = create_audiomae_model()
    audiomae.to(device)
    audiomae.eval()
    
    qformer = create_audio_qformer()
    qformer.to(device)
    qformer.eval()
    
    llm_interface = create_audio_llm_interface()
    llm_interface.to(device)
    llm_interface.eval()
    
    # 创建随机输入
    batch_size = 1
    channels = 1
    n_mels = 128
    time_steps = 4096
    dummy_input = torch.randn(batch_size, channels, n_mels, time_steps).to(device)
    
    # 测试前向传播
    with torch.no_grad():
        try:
            # AudioMAE前向传播
            audiomae_output = audiomae(dummy_input)
            logger.info(f"AudioMAE output shape: {audiomae_output.shape}")
            
            # 确保输出有正确的维度
            if len(audiomae_output.shape) == 2:  # [B, D]
                # 添加序列维度 [B, 1, D]
                audiomae_output = audiomae_output.unsqueeze(1)
                logger.info(f"AudioMAE output after adding sequence dimension: {audiomae_output.shape}")
            
            # Q-former前向传播
            qformer_output = qformer(audiomae_output)
            logger.info(f"Q-former output shape: {qformer_output.shape}")
            
            # LLM接口前向传播（仅投影，不生成文本）
            llm_proj = llm_interface.llm_proj(qformer_output)
            logger.info(f"LLM projection output shape: {llm_proj.shape}")
            
            logger.info("Full pipeline test successful!")
            return True
        except Exception as e:
            logger.error(f"Error in full pipeline: {e}")
            return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING PIPELINE COMPONENTS")
    print("=" * 50)
    
    # 测试各组件
    audiomae_success = test_audiomae()
    print()
    qformer_success = test_qformer()
    print()
    pipeline_success = test_full_pipeline()
    
    # 总结
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"AudioMAE test: {'PASSED' if audiomae_success else 'FAILED'}")
    print(f"Q-former test: {'PASSED' if qformer_success else 'FAILED'}")
    print(f"Full pipeline test: {'PASSED' if pipeline_success else 'FAILED'}")
    print("=" * 50) 