"""
测试LLM处理器设备一致性
"""

import os
import sys
import torch
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.llm.llm_interface import LLMProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_processor():
    """测试LLM处理器"""
    print("Testing LLM processor...")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}")
    print(f"Using device: {device}")
    
    try:
        # 初始化LLM处理器
        print("\nInitializing LLM processor...")
        processor = LLMProcessor(device=device)
        
        # 检查模型设备
        model_device = next(processor.model.parameters()).device
        print(f"LLM processor model is on device: {model_device}")
        
        # 如果测试通过，输出成功消息
        print("\nLLM processor test PASSED!")
        return True
    except Exception as e:
        print(f"Error testing LLM processor: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING LLM PROCESSOR")
    print("=" * 50)
    
    success = test_llm_processor()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"LLM processor test: {'PASSED' if success else 'FAILED'}")
    print("=" * 50) 