"""
训练钢琴演奏分析模型
"""

import os
import sys
import torch
import argparse
import logging
from datetime import datetime
import yaml

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.audio_trainer import PianoPerformanceTrainer

# 创建日志目录
os.makedirs("logs", exist_ok=True)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练钢琴演奏分析模型")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/audio_config.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="checkpoints",
        help="检查点保存目录"
    )
    
    parser.add_argument(
        "--log_dir", 
        type=str, 
        default="logs",
        help="日志保存目录"
    )
    
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="从检查点恢复训练的路径"
    )
    
    parser.add_argument(
        "--cpu", 
        action="store_true",
        help="强制使用CPU训练"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="训练轮数（覆盖配置文件中的设置）"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="批处理大小（覆盖配置文件中的设置）"
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=None,
        help="学习率（覆盖配置文件中的设置）"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 检查配置文件
    if not os.path.exists(args.config):
        logger.error(f"配置文件 {args.config} 不存在")
        return
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 覆盖配置文件中的设置（如果命令行参数提供）
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
        logger.info(f"覆盖训练轮数: {args.epochs}")
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        logger.info(f"覆盖批处理大小: {args.batch_size}")
    
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
        logger.info(f"覆盖学习率: {args.learning_rate}")
    
    # 保存更新后的配置
    with open(args.config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    # 确定设备
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建训练器
    trainer = PianoPerformanceTrainer(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=device
    )
    
    # 从检查点恢复（如果提供）
    if args.resume:
        if not os.path.exists(args.resume):
            logger.error(f"检查点 {args.resume} 不存在")
            return
        
        logger.info(f"从检查点 {args.resume} 恢复训练")
        if not trainer.load_checkpoint(args.resume):
            logger.error("恢复训练失败，退出")
            return
    
    # 打印训练配置
    logger.info(f"训练配置:")
    logger.info(f"- 配置文件: {args.config}")
    logger.info(f"- 训练轮数: {config['training']['num_epochs']}")
    logger.info(f"- 批处理大小: {config['training']['batch_size']}")
    logger.info(f"- 学习率: {config['training']['learning_rate']}")
    logger.info(f"- 梯度累积步数: {config['training']['gradient_accumulation_steps']}")
    
    # 开始训练
    logger.info("开始训练...")
    metrics_history = trainer.train()
    
    # 记录最终指标
    final_train_loss = metrics_history["train_loss"][-1]
    final_val_loss = metrics_history["val_loss"][-1]
    logger.info(f"训练完成!")
    logger.info(f"最终训练损失: {final_train_loss:.4f}")
    logger.info(f"最终验证损失: {final_val_loss:.4f}")
    
    return metrics_history

if __name__ == "__main__":
    main() 