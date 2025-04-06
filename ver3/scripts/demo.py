# scripts/demo.py

import os
import sys
import logging

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ver3.inference.feedback_generator import PianoFeedbackGenerator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo():
    """演示钢琴演奏分析系统的使用"""

    # 配置
    config_path = "config/audio_config.yaml"
    checkpoint_path = None  # 暂时不使用检查点

    # 初始化反馈生成器
    feedback_generator = PianoFeedbackGenerator(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device="cpu"  # 使用CPU进行演示
    )

    # 演示音频路径（替换为实际路径）
    audio_path = "data/demo_audio.wav"

    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        logger.error(f"演示音频文件不存在: {audio_path}")
        return

    # 生成反馈
    questions = [
        "How would you rate the rhythm stability of this performance?",
        "How accurate are the pitches in this performance?",
        "What's your overall feedback on this performance?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        feedback = feedback_generator.generate_feedback(audio_path, question)
        print(f"Feedback: {feedback}")

    print("\n生成全面的反馈...")
    comprehensive_feedback = feedback_generator.generate_comprehensive_feedback(audio_path)

    # 保存反馈
    output_path = "output/demo_feedback.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feedback_generator.save_feedback_to_json(comprehensive_feedback, output_path)

    print(f"\n全面反馈已保存到: {output_path}")


if __name__ == "__main__":
    demo()