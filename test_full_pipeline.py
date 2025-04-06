"""
测试完整模型管线
"""

import os
import sys
import torch
import logging
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion.multimodal_fusion import create_piano_performance_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def adjust_audiomae_position_embeddings(audiomae, input_size):
    """调整AudioMAE的位置嵌入以匹配输入尺寸"""
    n_mels, time_steps = input_size
    
    # 计算patch数量
    h_patches = n_mels // audiomae.patch_size[0]
    w_patches = time_steps // audiomae.patch_size[1]
    n_patches = h_patches * w_patches
    
    # 创建新的位置嵌入
    if hasattr(audiomae, 'pos_embed'):
        print(f"Original pos_embed shape: {audiomae.pos_embed.shape}")
        
        # 保存原始位置嵌入
        device = audiomae.pos_embed.device
        with torch.no_grad():
            original_pos_embed = audiomae.pos_embed.clone()
            
            # 创建新的位置嵌入
            embed_dim = audiomae.pos_embed.shape[2]
            new_pos_embed = torch.zeros(1, n_patches + 1, embed_dim, device=device)
            
            # 复制[CLS] token位置嵌入
            new_pos_embed[:, 0] = original_pos_embed[:, 0]
            
            # 剩余位置嵌入初始化为随机值
            new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
            
            # 应用新的位置嵌入
            audiomae.pos_embed = torch.nn.Parameter(new_pos_embed)
            print(f"Adjusted pos_embed shape: {audiomae.pos_embed.shape}")
            
        return original_pos_embed
    
    return None

def test_full_pipeline():
    """测试完整模型管线"""
    print("Testing full pipeline...")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}")
    print(f"Using device: {device}")
    
    # 读取配置
    config_path = "config/audio_config.yaml"
    print(f"\nReading config from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            audio_config = config['audio_model']
            n_mels = audio_config.get('n_mels', 128)
            max_length = 4096  # 硬编码为4096，匹配模型期望
            print(f"Using n_mels={n_mels}, max_length={max_length}")
    except Exception as e:
        print(f"Error reading config: {e}")
        n_mels = 128
        max_length = 4096
    
    try:
        # 创建完整模型
        print("\nCreating piano performance model...")
        model = create_piano_performance_model(device=device)
        
        # 检查模型组件
        print("\nModel components:")
        print(f"- AudioMAE on: {next(model.audio_encoder.parameters()).device}")
        print(f"- QFormer on: {next(model.qformer.parameters()).device}")
        print(f"- LLM Interface on: {next(model.llm_interface.parameters()).device}")
        
        # 创建一个测试输入
        print(f"\nCreating test input with shape: [1, 1, {n_mels}, {max_length}]")
        batch_size = 1
        dummy_input = torch.randn(batch_size, 1, n_mels, max_length, device=device)
        dummy_question = "What's good about this performance?"
        
        # 执行前向传播
        print("\nPerforming forward pass without generation...")
        with torch.no_grad():
            model.eval()
            
            # 1. AudioMAE处理
            print("Running audio_encoder forward pass...")
            audio_features, _ = model.audio_encoder(dummy_input, return_patch_features=True)
            print(f"AudioMAE output: {audio_features.shape} on {audio_features.device}")
            
            # 2. QFormer处理
            print("Running qformer forward pass...")
            query_embeds = model.qformer(audio_features)
            print(f"QFormer output: {query_embeds.shape} on {query_embeds.device}")
            
            # 3. LLM投影层
            print("Running llm_interface.llm_proj forward pass...")
            projected_embeds = model.llm_interface.llm_proj(query_embeds)
            print(f"LLM projection output: {projected_embeds.shape} on {projected_embeds.device}")
            
            print("\nAll components initialized and tested successfully!")
        
        # 测试使用1024时间步的情况 (需要调整位置嵌入)
        print("\n" + "=" * 50)
        print("TESTING WITH SHORTER INPUT (1024 TIME STEPS)")
        print("=" * 50)
        
        # 创建较短的输入
        short_length = 1024
        print(f"\nCreating shorter test input with shape: [1, 1, {n_mels}, {short_length}]")
        short_input = torch.randn(batch_size, 1, n_mels, short_length, device=device)
        
        # 调整位置嵌入
        print("\nAdjusting position embeddings for shorter input...")
        original_pos_embed = adjust_audiomae_position_embeddings(
            model.audio_encoder, 
            input_size=(n_mels, short_length)
        )
        
        # 执行前向传播
        print("\nPerforming forward pass with shorter input...")
        with torch.no_grad():
            model.eval()
            
            # 1. AudioMAE处理
            print("Running audio_encoder forward pass...")
            audio_features, _ = model.audio_encoder(short_input, return_patch_features=True)
            print(f"AudioMAE output: {audio_features.shape} on {audio_features.device}")
            
            # 2. QFormer处理
            print("Running qformer forward pass...")
            query_embeds = model.qformer(audio_features)
            print(f"QFormer output: {query_embeds.shape} on {query_embeds.device}")
            
            # 3. LLM投影层
            print("Running llm_interface.llm_proj forward pass...")
            projected_embeds = model.llm_interface.llm_proj(query_embeds)
            print(f"LLM projection output: {projected_embeds.shape} on {projected_embeds.device}")
            
            print("\nAll components work with shorter input after position embedding adjustment!")
            
        # 恢复原始位置嵌入
        if original_pos_embed is not None:
            with torch.no_grad():
                model.audio_encoder.pos_embed = torch.nn.Parameter(original_pos_embed)
                print("\nRestored original position embeddings.")
        
        return True
    except Exception as e:
        print(f"Error in full pipeline test: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("TESTING FULL MODEL PIPELINE")
    print("=" * 50)
    
    success = test_full_pipeline()
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Full pipeline test: {'PASSED' if success else 'FAILED'}")
    print("=" * 50) 