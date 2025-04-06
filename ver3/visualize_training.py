"""
可视化训练进度
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import json
from datetime import datetime

def plot_loss_curves(
        train_loss,
        val_loss,
        save_path=None,
        title="Training and Validation Loss",
        figsize=(10, 6)
):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_loss: 训练损失列表
        val_loss: 验证损失列表
        save_path: 保存路径（可选）
        title: 图表标题
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    epochs = np.arange(1, len(train_loss) + 1)
    
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 找到最小验证损失并标记
    min_val_idx = np.argmin(val_loss)
    min_val = val_loss[min_val_idx]
    plt.plot(min_val_idx + 1, min_val, 'ro', markersize=8)
    plt.annotate(f'min: {min_val:.4f}', 
                xy=(min_val_idx + 1, min_val),
                xytext=(min_val_idx + 1 + 0.5, min_val + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    plt.show()

def plot_lr_curve(learning_rates, save_path=None, title="Learning Rate Schedule", figsize=(10, 6)):
    """
    绘制学习率曲线
    
    Args:
        learning_rates: 学习率列表
        save_path: 保存路径（可选）
        title: 图表标题
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    steps = np.arange(1, len(learning_rates) + 1)
    
    plt.plot(steps, learning_rates, 'g-')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 使用对数刻度以更好地显示学习率变化
    plt.yscale('log')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到 {save_path}")
    
    plt.show()

def load_metrics(metrics_path):
    """加载指标历史"""
    with open(metrics_path, 'r') as f:
        if metrics_path.endswith('.json'):
            metrics = json.load(f)
        else:  # 假设是PyTorch保存的文件
            metrics = torch.load(metrics_path)
    return metrics

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="可视化训练进度")
    
    parser.add_argument(
        "--metrics", 
        type=str, 
        required=True,
        help="指标历史文件路径"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="visualizations",
        help="输出目录"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载指标
    try:
        metrics = load_metrics(args.metrics)
    except Exception as e:
        print(f"加载指标文件时出错: {e}")
        return
    
    # 检查所需的指标是否存在
    required_metrics = ["train_loss", "val_loss"]
    for metric in required_metrics:
        if metric not in metrics:
            print(f"错误: 指标文件中缺少 '{metric}'")
            return
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 绘制损失曲线
    loss_plot_path = os.path.join(args.output_dir, f"loss_curves_{timestamp}.png")
    plot_loss_curves(
        metrics["train_loss"],
        metrics["val_loss"],
        save_path=loss_plot_path,
        title="训练和验证损失"
    )
    
    # 如果有学习率数据，绘制学习率曲线
    if "lr" in metrics:
        lr_plot_path = os.path.join(args.output_dir, f"lr_curve_{timestamp}.png")
        plot_lr_curve(
            metrics["lr"],
            save_path=lr_plot_path,
            title="学习率变化"
        )
    
    print("可视化完成！")

if __name__ == "__main__":
    main() 