# models/fusion/multimodal_fusion.py

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 修复导入路径，移除 ver3. 前缀
from models.audio.audiomae import create_audiomae_model, AudioMAE
from models.fusion.qformer import create_audio_qformer, AudioQFormer
from models.llm.llm_interface import create_audio_llm_interface, AudioLLMInterface

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PianoPerformanceModel(nn.Module):
    """
    钢琴演奏分析模型，整合音频分析和LLM反馈生成
    """

    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化钢琴演奏分析模型

        Args:
            config_path: 配置文件路径
            device: 计算设备
        """
        super().__init__()
        self.config_path = config_path
        self.device = device

        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, creating default config")
            # 创建默认配置
            config = {
                'audio_model': {
                    'model_type': 'audiomae',
                    'pretrained_path': 'pretrained/audiomae_base.pth',
                    'sample_rate': 32000,
                    'n_mels': 128,
                    'feature_dim': 1024,
                    'max_length': 1024,
                    'frameshift': 10
                }
            }
            # 写入配置文件
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)

        # 创建模型组件
        self.audio_encoder = create_audiomae_model(config_path).to(device)
        self.qformer = create_audio_qformer(config_path).to(device)
        self.llm_interface = create_audio_llm_interface(config_path).to(device)

        logger.info(f"Initialized PianoPerformanceModel on {device}")

    def forward(
            self,
            mel_spectrogram: torch.Tensor,
            question: str,
            max_new_tokens: int = 50,
            temperature: float = 1.0
    ) -> List[str]:
        """
        模型前向传播

        Args:
            mel_spectrogram: Mel频谱图 [B, 1, n_mels, T]
            question: 问题文本
            max_new_tokens: 生成的最大新令牌数
            temperature: 生成温度

        Returns:
            生成的回答列表
        """
        # 1. 通过AudioMAE提取音频特征
        with torch.no_grad():
            audio_features, _ = self.audio_encoder(mel_spectrogram, return_patch_features=True)

        # 2. 使用Q-former提取查询特征
        query_embeds = self.qformer(audio_features)

        # 3. 通过LLM接口生成回答
        answers = self.llm_interface(
            query_embeds=query_embeds,
            question=question,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        return answers

    def generate_feedback(
            self,
            audio_path: str,
            question: str,
            audio_processor=None,
            max_new_tokens: int = 100,
            temperature: float = 0.7
    ) -> str:
        """
        为钢琴演奏生成反馈

        Args:
            audio_path: 音频文件路径
            question: 问题文本
            audio_processor: 音频处理器实例（可选）
            max_new_tokens: 生成的最大新令牌数
            temperature: 生成温度

        Returns:
            生成的反馈
        """
        # 确保有音频处理器
        if audio_processor is None:
            from data.preprocessing.audio_processor import AudioProcessor
            audio_processor = AudioProcessor(self.config_path)

        # 处理音频
        audio_features = audio_processor.process_audio(audio_path)
        mel_spectrogram = audio_features["mel_spectrogram"].unsqueeze(0).to(self.device)  # 添加批次维度

        # 设置为评估模式
        self.eval()

        # 生成反馈
        with torch.no_grad():
            answers = self.forward(
                mel_spectrogram=mel_spectrogram,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )

        # 返回第一个（也是唯一的）回答
        return answers[0] if answers else "无法生成反馈。"


def create_piano_performance_model(
        config_path: str = "config/audio_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> PianoPerformanceModel:
    """
    创建钢琴演奏分析模型

    Args:
        config_path: 配置文件路径
        device: 计算设备

    Returns:
        PianoPerformanceModel实例
    """
    return PianoPerformanceModel(config_path, device)


# 测试代码
if __name__ == "__main__":
    # 创建配置文件目录（如果不存在）
    os.makedirs("config", exist_ok=True)

    # 如果配置文件不存在，创建一个临时配置
    if not os.path.exists("config/audio_config.yaml"):
        print("创建临时配置文件...")
        # 这里我们会使用前面已经定义的配置文件格式

    try:
        # 在CPU上创建模型（用于测试）
        print("创建模型...")
        model = create_piano_performance_model(device="cpu")
        print(f"模型创建成功: {type(model).__name__}")

        # 你可以在这里添加更多测试，但由于需要完整的模型权重，
        # 简单的模型实例化测试可能更合适
    except Exception as e:
        print(f"模型创建错误: {e}")