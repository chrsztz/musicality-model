"""
测试AudioMAE位置嵌入调整
"""

import os
import sys
import torch
import logging
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio.audiomae import create_audiomae_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_audiomae_positions():
    """测试AudioMAE位置嵌入调整"""
    print("Testing AudioMAE with position embedding adjustment...")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}")
    print(f"Using device: {device}")
    
    # 首先打印配置文件中的值
    config_path = "config/audio_config.yaml"
    print(f"\nReading config from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            audio_config = config['audio_model']
            print(f"Config audio_model.max_length: {audio_config.get('max_length')}")
            print(f"Config audio_model.n_mels: {audio_config.get('n_mels')}")
            # 直接从配置获取参数用于创建输入
            n_mels = audio_config.get('n_mels', 128)
            max_length = audio_config.get('max_length', 4096)
            print(f"Using n_mels={n_mels}, max_length={max_length} from config")
    except Exception as e:
        print(f"Error reading config: {e}")
        n_mels = 128
        max_length = 4096
        print(f"Using fallback values: n_mels={n_mels}, max_length={max_length}")
    
    try:
        # 创建AudioMAE模型
        print("\nCreating AudioMAE model...")
        audiomae = create_audiomae_model(config_path)
        audiomae = audiomae.to(device)
        
        # 打印模型参数
        print(f"- AudioMAE img_size: {audiomae.img_size}")
        print(f"- AudioMAE patch_size: {audiomae.patch_size}")
        print(f"- AudioMAE patch_embed.img_size: {audiomae.patch_embed.img_size}")
        
        # 创建测试输入 - 故意使用较小的时间步长
        print("\nCreating test input with max_length=1024 (smaller than model expects)...")
        batch_size = 1
        test_length = 1024  # 故意使用1024而不是4096，模拟问题情况
        
        dummy_input = torch.randn(batch_size, 1, n_mels, test_length, device=device)
        print(f"Input shape: {dummy_input.shape}")
        
        # 计算实际的patch数量
        h_patches = n_mels // audiomae.patch_size[0]  # 频率维度的patch数
        w_patches = test_length // audiomae.patch_size[1]  # 时间维度的patch数
        n_patches = h_patches * w_patches
        print(f"Expected patches: {n_patches} (= {h_patches} * {w_patches})")
        
        # 检查pos_embed的大小
        if hasattr(audiomae, 'pos_embed'):
            print(f"Current pos_embed shape: {audiomae.pos_embed.shape}")
            
            # 创建新的位置嵌入，大小与实际的patch数匹配
            print("\nCreating new position embeddings to match input size...")
            with torch.no_grad():
                # 保存原始的位置嵌入，以备需要
                original_pos_embed = audiomae.pos_embed.clone()
                
                # 创建新的位置嵌入，大小与实际的patch数匹配
                # 通常位置嵌入的形状是 [1, num_patches+1, embed_dim] 
                # 第一个token是 [CLS] token
                embed_dim = audiomae.pos_embed.shape[2]
                new_pos_embed = torch.zeros(1, n_patches + 1, embed_dim, device=device)
                
                # 复制 [CLS] token 的位置嵌入
                new_pos_embed[:, 0] = original_pos_embed[:, 0]
                
                # 简单方法：将其余位置嵌入初始化为随机值
                # 更好的方法是从原始的位置嵌入插值，但这里简化处理
                new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
                
                # 替换模型中的位置嵌入
                audiomae.pos_embed = torch.nn.Parameter(new_pos_embed)
                print(f"New pos_embed shape: {audiomae.pos_embed.shape}")
        
        # 执行前向传播
        print("\nPerforming forward pass with adjusted position embeddings...")
        with torch.no_grad():
            audiomae.eval()
            features, pooled = audiomae(dummy_input, return_patch_features=True)
            print(f"Patch features shape: {features.shape}")
            print(f"Pooled features shape: {pooled.shape}")
            print("Forward pass successful with adjusted position embeddings!")
            
        # 现在尝试标准的 4096 长度输入
        print("\nCreating test input with standard max_length=4096...")
        standard_input = torch.randn(batch_size, 1, n_mels, max_length, device=device)
        print(f"Standard input shape: {standard_input.shape}")
        
        # 还原原始位置嵌入
        with torch.no_grad():
            audiomae.pos_embed = torch.nn.Parameter(original_pos_embed)
            print(f"Restored original pos_embed shape: {audiomae.pos_embed.shape}")
        
        # 执行前向传播
        print("\nPerforming forward pass with standard input and original position embeddings...")
        with torch.no_grad():
            audiomae.eval()
            features, pooled = audiomae(standard_input, return_patch_features=True)
            print(f"Patch features shape: {features.shape}")
            print(f"Pooled features shape: {pooled.shape}")
            print("Forward pass successful with standard input!")
        
        return True
    except Exception as e:
        print(f"Error in AudioMAE position test: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING AUDIOMAE POSITION EMBEDDINGS")
    print("=" * 50)
    
    success = test_audiomae_positions()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"AudioMAE position embedding test: {'PASSED' if success else 'FAILED'}")
    print("=" * 50) 