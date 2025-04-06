"""
测试已训练的钢琴演奏分析模型
"""

import os
import sys
import torch
import argparse
import logging
import yaml
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.fusion.multimodal_fusion import create_piano_performance_model
from data.preprocessing.audio_processor import AudioProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_from_checkpoint(
        checkpoint_path: str,
        config_path: str = "config/audio_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> torch.nn.Module:
    """
    从检查点加载模型
    
    Args:
        checkpoint_path: 检查点路径
        config_path: 配置文件路径
        device: 设备
    
    Returns:
        加载的模型
    """
    # 创建模型
    model = create_piano_performance_model(config_path, device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型状态
    model.qformer.load_state_dict(checkpoint["qformer_state_dict"])
    model.llm_interface.llm_proj.load_state_dict(checkpoint["llm_proj_state_dict"])
    
    logger.info(f"成功从 {checkpoint_path} 加载模型")
    return model.to(device)

def process_audio_file(
        model: torch.nn.Module,
        audio_processor: AudioProcessor,
        audio_path: str,
        question: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> str:
    """
    处理音频文件并生成回答
    
    Args:
        model: 模型
        audio_processor: 音频处理器
        audio_path: 音频文件路径
        question: 问题
        device: 设备
    
    Returns:
        模型的回答
    """
    # 处理音频
    audio_features = audio_processor.process_audio(audio_path)
    mel_spectrogram = audio_features["mel_spectrogram"].unsqueeze(0).to(device)  # 添加批次维度
    
    # 检查输入形状
    logger.info(f"输入形状: {mel_spectrogram.shape}")
    batch_size, channels, n_mels, time_steps = mel_spectrogram.shape
    
    # 如果时间步长不是4096，调整位置嵌入
    original_pos_embed = None
    if time_steps != 4096 and hasattr(model.audio_encoder, 'pos_embed'):
        # 保存原始位置嵌入
        original_pos_embed = model.audio_encoder.pos_embed.clone()
        
        # 计算新的patch数量
        h_patches = n_mels // model.audio_encoder.patch_size[0]
        w_patches = time_steps // model.audio_encoder.patch_size[1]
        n_patches = h_patches * w_patches
        
        # 创建新的位置嵌入
        embed_dim = model.audio_encoder.pos_embed.shape[2]
        new_pos_embed = torch.zeros(1, n_patches, embed_dim, device=device)
        
        # 复制CLS token位置嵌入
        new_pos_embed[:, 0] = original_pos_embed[:, 0]
        
        # 初始化其余部分为随机值
        new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
        
        # 应用新的位置嵌入
        model.audio_encoder.pos_embed = torch.nn.Parameter(new_pos_embed)
    
    # 设置为评估模式
    model.eval()
    
    # 生成回答
    with torch.no_grad():
        try:
            # 执行部分前向传播
            audio_features, _ = model.audio_encoder(mel_spectrogram, return_patch_features=True)
            logger.info(f"AudioMAE输出形状: {audio_features.shape}")
            
            query_embeds = model.qformer(audio_features)
            logger.info(f"QFormer输出形状: {query_embeds.shape}")
            
            # 分两种情况：如果模型的forward方法已经实现了完整逻辑，使用它；
            # 否则，使用llm_interface进行生成
            if hasattr(model, 'forward') and callable(model.forward):
                answer = model.forward(mel_spectrogram, question, max_new_tokens=100, temperature=0.7)
                if isinstance(answer, list):
                    answer = answer[0]
            else:
                answers = model.llm_interface(
                    query_embeds=query_embeds,
                    question=question,
                    max_new_tokens=100,
                    temperature=0.7
                )
                answer = answers[0] if answers else "无法生成回答。"
        except Exception as e:
            logger.error(f"生成回答时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            answer = f"发生错误: {str(e)}"
    
    # 恢复原始位置嵌入
    if original_pos_embed is not None:
        model.audio_encoder.pos_embed = torch.nn.Parameter(original_pos_embed)
    
    return answer

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试已训练的钢琴演奏分析模型")
    
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="模型检查点路径"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/audio_config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--audio", 
        type=str, 
        required=True,
        help="音频文件路径"
    )
    
    parser.add_argument(
        "--question", 
        type=str, 
        default="请分析这个钢琴演奏的音乐特点。",
        help="要问的问题"
    )
    
    parser.add_argument(
        "--cpu", 
        action="store_true",
        help="强制使用CPU"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.checkpoint):
        logger.error(f"检查点 {args.checkpoint} 不存在")
        return
    
    if not os.path.exists(args.config):
        logger.error(f"配置文件 {args.config} 不存在")
        return
    
    if not os.path.exists(args.audio):
        logger.error(f"音频文件 {args.audio} 不存在")
        return
    
    # 确定设备
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建音频处理器
    audio_processor = AudioProcessor(config_path=args.config)
    
    # 加载模型
    model = load_model_from_checkpoint(args.checkpoint, args.config, device)
    
    # 处理音频并生成回答
    logger.info(f"处理音频: {args.audio}")
    logger.info(f"问题: {args.question}")
    answer = process_audio_file(model, audio_processor, args.audio, args.question, device)
    
    # 打印结果
    print("\n" + "=" * 50)
    print("问题:", args.question)
    print("-" * 50)
    print("回答:", answer)
    print("=" * 50)
    
    return answer

if __name__ == "__main__":
    main() 