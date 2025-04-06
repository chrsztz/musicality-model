"""
测试HuggingFace NeuroPiano数据集的加载
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from data.datasets.neuropiano_hf import NeuroPianoHFDataset

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_raw_hf_dataset():
    """测试直接从HuggingFace加载原始数据集"""
    try:
        print("Attempting to load NeuroPiano dataset directly from HuggingFace...")
        dataset = load_dataset("anusfoil/NeuroPiano-data", split="train")
        print(f"Successfully loaded dataset with {len(dataset)} entries")
        
        # 显示一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample data:")
            for key, value in sample.items():
                if key == 'audio_path':
                    print(f"  {key}: <audio data type: {type(value)}>")
                    if hasattr(value, 'shape'):
                        print(f"    shape: {value.shape}")
                else:
                    # 尝试打印值，如果是字符串，可能需要编码处理
                    try:
                        print(f"  {key}: {value}")
                    except UnicodeEncodeError:
                        print(f"  {key}: <content contains non-ASCII characters>")
        return True
    except Exception as e:
        logger.error(f"Error loading raw HuggingFace dataset: {e}")
        return False

def test_neuropiano_hf_dataset():
    """测试我们的NeuroPianoHFDataset封装类"""
    try:
        print("\nAttempting to load dataset using NeuroPianoHFDataset...")
        
        # 确保配置目录存在
        os.makedirs("config", exist_ok=True)
        
        # 创建数据集实例，使用更小的subset来测试
        dataset = NeuroPianoHFDataset(
            split="train",
            max_qa_pairs=2  # 仅用于测试，限制加载的样本数
        )
        
        print(f"Successfully created NeuroPianoHFDataset with {len(dataset)} entries")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            try:
                sample = dataset[0]
                print("\nSample from NeuroPianoHFDataset:")
                print(f"  piece: {sample['piece']}")
                print(f"  subject: {sample['subject']}")
                if 'mel_spectrogram' in sample:
                    print(f"  mel_spectrogram shape: {sample['mel_spectrogram'].shape}")
                print(f"  question: {sample['question']}")
                
                # 尝试打印答案，处理可能的编码问题
                try:
                    answer_preview = sample['answer'][:100] + "..." if len(sample['answer']) > 100 else sample['answer']
                    print(f"  answer: {answer_preview}")
                except UnicodeEncodeError:
                    print("  answer: <content contains non-ASCII characters>")
                
                # 查看问题类型
                question_types = dataset.get_question_types()
                print("\nQuestion types:")
                for q_type, count in list(question_types.items())[:3]:  # 只显示前3种
                    print(f"  - {q_type}: {count}")
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                print(f"Error details: {e}")
        return True
    except Exception as e:
        logger.error(f"Error testing NeuroPianoHFDataset: {e}")
        print(f"Error details: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING NEUROPIANO HUGGINGFACE DATASET")
    print("=" * 50)
    
    # 测试原始HuggingFace数据集
    raw_success = test_raw_hf_dataset()
    
    # 测试我们的封装类
    hf_success = test_neuropiano_hf_dataset()
    
    # 总结
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Raw HuggingFace dataset test: {'PASSED' if raw_success else 'FAILED'}")
    print(f"NeuroPianoHFDataset test: {'PASSED' if hf_success else 'FAILED'}")
    print("=" * 50) 