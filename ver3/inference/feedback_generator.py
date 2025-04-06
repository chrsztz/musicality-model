# inference/feedback_generator.py

import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import sys
import time
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ver3.models.fusion.multimodal_fusion import create_piano_performance_model
from ver3.data.preprocessing.audio_processor import AudioProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PianoFeedbackGenerator:
    """
    钢琴演奏反馈生成器，用于推理
    """

    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            checkpoint_path: Optional[str] = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化反馈生成器

        Args:
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径（如果有）
            device: 计算设备
        """
        self.config_path = config_path
        self.device = device

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 创建音频处理器
        self.audio_processor = AudioProcessor(config_path)

        # 创建模型
        self.model = create_piano_performance_model(config_path, device)

        # 加载检查点（如果有）
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)

        # 设置为评估模式
        self.model.eval()

        logger.info(f"Initialized PianoFeedbackGenerator on {device}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 恢复模型状态
            self.model.qformer.load_state_dict(checkpoint["qformer_state_dict"])
            self.model.llm_interface.llm_proj.load_state_dict(checkpoint["llm_proj_state_dict"])

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")

    def generate_feedback(
            self,
            audio_path: str,
            question: str,
            max_new_tokens: int = 100,
            temperature: float = 0.7,
            num_beams: int = 1,
            top_p: float = 0.9,
            do_sample: bool = True
    ) -> str:
        """
        为钢琴演奏生成反馈

        Args:
            audio_path: 音频文件路径
            question: 问题文本
            max_new_tokens: 生成的最大新令牌数
            temperature: 生成温度
            num_beams: 光束搜索的光束数
            top_p: 核采样概率
            do_sample: 是否使用采样

        Returns:
            生成的反馈
        """
        start_time = time.time()

        # 处理音频
        audio_features = self.audio_processor.process_audio(audio_path)
        mel_spectrogram = audio_features["mel_spectrogram"].unsqueeze(0).to(self.device)  # 添加批次维度

        # 生成反馈
        with torch.no_grad():
            # 1. 通过AudioMAE提取音频特征
            audio_features, _ = self.model.audio_encoder(mel_spectrogram, return_patch_features=True)

            # 2. 使用Q-former提取查询特征
            query_embeds = self.model.qformer(audio_features)

            # 3. 通过LLM接口生成回答
            answers = self.model.llm_interface(
                query_embeds=query_embeds,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
                top_p=top_p,
                do_sample=do_sample
            )

        end_time = time.time()
        processing_time = end_time - start_time

        logger.info(f"Generated feedback in {processing_time:.2f} seconds")

        # 返回第一个（也是唯一的）回答
        return answers[0] if answers else "无法生成反馈。"

    def generate_comprehensive_feedback(self, audio_path: str) -> Dict[str, str]:
        """
        生成全面的钢琴演奏反馈，包括多个维度

        Args:
            audio_path: 音频文件路径

        Returns:
            包含各个维度反馈的字典
        """
        # 定义评估维度和对应的问题
        dimensions = {
            "pitch_accuracy": "How accurate are the pitches in this performance?",
            "rhythm_stability": "How would you rate the rhythm stability of this performance?",
            "tempo_consistency": "Is the tempo consistent throughout the performance?",
            "articulation": "How would you describe the articulation in this performance?",
            "dynamics": "How well are dynamics handled in this performance?",
            "expression": "How expressive is this performance?",
            "overall": "What's your overall feedback on this performance?"
        }

        # 为每个维度生成反馈
        feedback = {}
        for dim, question in dimensions.items():
            logger.info(f"Generating feedback for dimension: {dim}")
            feedback[dim] = self.generate_feedback(audio_path, question)

        return feedback

    def save_feedback_to_json(self, feedback: Dict[str, str], output_path: str):
        """
        将反馈保存为JSON文件

        Args:
            feedback: 反馈字典
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(feedback, f, ensure_ascii=False, indent=2)

        logger.info(f"Feedback saved to {output_path}")


# 测试代码
if __name__ == "__main__":
    # 测试反馈生成器
    feedback_generator = PianoFeedbackGenerator(device="cpu")
    print(f"Feedback generator initialized: {type(feedback_generator).__name__}")

    # 你可以添加一个测试音频文件来测试反馈生成
    test_audio_path = "path/to/test_audio.wav"  # 替换为实际测试音频路径

    if os.path.exists(test_audio_path):
        # 生成单个反馈
        question = "How would you rate the rhythm stability of this performance?"
        feedback = feedback_generator.generate_feedback(test_audio_path, question)
        print(f"Question: {question}")
        print(f"Feedback: {feedback}")

        # 生成全面的反馈
        comprehensive_feedback = feedback_generator.generate_comprehensive_feedback(test_audio_path)
        for dim, fb in comprehensive_feedback.items():
            print(f"{dim}: {fb[:100]}...")  # 只显示前100个字符
    else:
        print(f"Test audio file not found: {test_audio_path}")