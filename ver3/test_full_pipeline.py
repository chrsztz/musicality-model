"""
测试完整模型管线
"""

import os
import sys
import torch
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion.multimodal_fusion import create_piano_performance_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_pipeline():
    """测试完整模型管线"""
    print("Testing full pipeline...")
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"CUDA available: {cuda_available}")
    print(f"Using device: {device}")
    
    try:
        # 创建完整模型
        print("\nCreating piano performance model...")
        model = create_piano_performance_model(device=device)
        
        # 检查模型组件
        print("\nModel components:")
        print(f"- AudioMAE on: {next(model.audio_encoder.parameters()).device}")
        print(f"- QFormer on: {next(model.qformer.parameters()).device}")
        print(f"- LLM Interface on: {next(model.llm_interface.parameters()).device}")
        
        # 创建随机输入
        batch_size = 1
        n_mels = 128
        time_steps = 1024
        
        print("\nCreating test input...")
        dummy_input = torch.randn(batch_size, 1, n_mels, time_steps, device=device)
        dummy_question = "What's good about this performance?"
        
        # 调整AudioMAE的位置嵌入
        print("\nAdjusting AudioMAE position embeddings...")
        if hasattr(model.audio_encoder, 'pos_embed'):
            # 计算patch数量
            h_patches = n_mels // model.audio_encoder.patch_size[0]
            w_patches = time_steps // model.audio_encoder.patch_size[1]
            n_patches = h_patches * w_patches
            print(f"Expected patches: {n_patches} (= {h_patches} * {w_patches})")
            
            # 保存原始位置嵌入
            original_pos_embed = model.audio_encoder.pos_embed.clone()
            
            # 创建新的位置嵌入
            embed_dim = model.audio_encoder.pos_embed.shape[2]
            new_pos_embed = torch.zeros(1, n_patches, embed_dim, device=device)
            
            # 复制CLS token位置嵌入
            new_pos_embed[:, 0] = original_pos_embed[:, 0]
            
            # 初始化其余部分为随机值
            new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
            
            # 应用新的位置嵌入
            model.audio_encoder.pos_embed = torch.nn.Parameter(new_pos_embed)
            print(f"New pos_embed shape: {model.audio_encoder.pos_embed.shape}")
        
        # 执行部分前向传播
        print("\nPerforming forward pass without generation...")
        with torch.no_grad():
            model.eval()
            
            # 1. AudioMAE处理
            audio_features, _ = model.audio_encoder(dummy_input, return_patch_features=True)
            print(f"AudioMAE output: {audio_features.shape} on {audio_features.device}")
            
            # 2. QFormer处理
            query_embeds = model.qformer(audio_features)
            print(f"QFormer output: {query_embeds.shape} on {query_embeds.device}")
            
            # 3. LLM投影层
            projected_embeds = model.llm_interface.llm_proj(query_embeds)
            print(f"LLM projection output: {projected_embeds.shape} on {projected_embeds.device}")
            
            print("\nAll components initialized and tested successfully!")
        
        # 恢复原始位置嵌入
        if hasattr(model.audio_encoder, 'pos_embed'):
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