# main.py

import os
import argparse
import logging
import yaml
import torch

from training.audio_trainer import PianoPerformanceTrainer
from inference.feedback_generator import PianoFeedbackGenerator
from models.fusion.multimodal_fusion import create_piano_performance_model
from data.preprocessing.audio_processor import AudioProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description="钢琴演奏分析AI系统")

    # 通用参数
    parser.add_argument("--config", type=str, default="config/audio_config.yaml", help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="计算设备")

    # 模式选择
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # 训练模式参数
    train_parser = subparsers.add_parser("train", help="训练模式")
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="检查点保存目录")
    train_parser.add_argument("--log_dir", type=str, default="logs", help="日志保存目录")
    train_parser.add_argument("--resume_checkpoint", type=str, default=None, help="恢复训练的检查点路径")

    # 推理模式参数
    infer_parser = subparsers.add_parser("infer", help="推理模式")
    infer_parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    infer_parser.add_argument("--audio", type=str, required=True, help="音频文件路径")
    infer_parser.add_argument("--question", type=str, default="What's your overall feedback on this performance?",
                              help="问题文本")
    infer_parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    infer_parser.add_argument("--comprehensive", action="store_true", help="生成全面的反馈")

    # 解析参数
    args = parser.parse_args()

    # 确保配置文件存在
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        return

    # 根据模式执行相应操作
    if args.mode == "train":
        # 训练模式
        logger.info("启动训练模式")
        trainer = PianoPerformanceTrainer(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir,
            device=args.device
        )

        # 恢复检查点（如果有）
        if args.resume_checkpoint:
            trainer.load_checkpoint(args.resume_checkpoint)

        # 开始训练
        trainer.train()

    elif args.mode == "infer":
        # 推理模式
        logger.info("启动推理模式")

        # 检查音频文件是否存在
        if not os.path.exists(args.audio):
            logger.error(f"音频文件不存在: {args.audio}")
            return

        # 创建反馈生成器
        feedback_generator = PianoFeedbackGenerator(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )

        # 生成反馈
        if args.comprehensive:
            # 生成全面的反馈
            feedback = feedback_generator.generate_comprehensive_feedback(args.audio)

            # 打印反馈
            for dim, fb in feedback.items():
                print(f"\n{dim.upper()}:")
                print(fb)

            # 保存到文件（如果指定）
            if args.output:
                feedback_generator.save_feedback_to_json(feedback, args.output)
                logger.info(f"全面反馈已保存到: {args.output}")
        else:
            # 生成单个反馈
            feedback = feedback_generator.generate_feedback(args.audio, args.question)

            # 打印反馈
            print(f"\nQuestion: {args.question}")
            print(f"Feedback: {feedback}")

            # 保存到文件（如果指定）
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(feedback)
                logger.info(f"反馈已保存到: {args.output}")

    else:
        # 未指定模式
        parser.print_help()


if __name__ == "__main__":
    main()