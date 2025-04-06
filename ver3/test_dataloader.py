"""
测试钢琴演奏数据加载器
"""

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataloader import PianoDataLoader

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dataloader():
    """测试PianoDataLoader"""
    try:
        print("Testing PianoDataLoader...")
        
        # 确保配置目录存在
        os.makedirs("config", exist_ok=True)
        
        # 创建数据加载器实例
        dataloader = PianoDataLoader(
            batch_size=2,  # 小批量用于测试
            num_workers=0  # 单线程进行调试
        )
        
        # 获取数据加载器
        train_loader = dataloader.get_train_dataloader()
        val_loader = dataloader.get_val_dataloader()
        test_loader = dataloader.get_test_dataloader()
        
        print(f"Created data loaders:")
        print(f"  - Train: {len(train_loader)} batches")
        print(f"  - Validation: {len(val_loader)} batches")
        print(f"  - Test: {len(test_loader)} batches")
        
        # 获取并检查一个小批量
        print("\nFetching a sample batch from training data...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  - Batch size: {len(batch['mel_spectrogram'])}")
            print(f"  - Mel spectrogram shape: {batch['mel_spectrogram'].shape}")
            print(f"  - Question: {batch['question'][0]}")
            print(f"  - Answer: {batch['answer'][0][:100]}..." if len(batch['answer'][0]) > 100 else batch['answer'][0])
            
            # 仅检查第一个批次
            break
        
        return True
    except Exception as e:
        logger.error(f"Error testing dataloader: {e}")
        print(f"Error details: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING PIANO PERFORMANCE DATALOADER")
    print("=" * 50)
    
    dataloader_success = test_dataloader()
    
    # 总结
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"PianoDataLoader test: {'PASSED' if dataloader_success else 'FAILED'}")
    print("=" * 50) 