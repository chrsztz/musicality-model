"""
运行完整测试流程

该脚本将执行所有必要的步骤来测试钢琴演奏分析模型
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行钢琴演奏分析模型测试')
    parser.add_argument(
        '--model', 
        type=str, 
        default="Qwen/QwQ-32B",
        help='要使用的LLM模型名称'
    )
    parser.add_argument(
        '--skip_download', 
        action='store_true',
        help='跳过模型下载步骤'
    )
    parser.add_argument(
        '--test_pipeline', 
        action='store_true',
        help='测试完整模型管线'
    )
    parser.add_argument(
        '--test_audiomae', 
        action='store_true',
        help='测试AudioMAE模型'
    )
    parser.add_argument(
        '--test_qformer', 
        action='store_true',
        help='测试Q-former模型'
    )
    return parser.parse_args()

def run_command(command, title=None):
    """运行命令并打印输出"""
    if title:
        print("\n" + "=" * 60)
        print(f" {title} ".center(60, "="))
        print("=" * 60 + "\n")
    
    print(f"执行命令: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    return process.returncode

def fix_encoding():
    """修复编码问题"""
    return run_command("python fix_encoding.py", "修复编码问题")

def download_model(model_name):
    """下载LLM模型"""
    return run_command(f"python download_llm.py --model {model_name}", "下载LLM模型")

def test_audiomae():
    """测试AudioMAE模型"""
    return run_command("python test_audiomae.py", "测试AudioMAE模型")

def test_qformer():
    """测试Q-former模型"""
    return run_command("python test_qformer.py", "测试Q-former模型")

def test_pipeline():
    """测试完整模型管线"""
    return run_command("python test_model_pipeline.py", "测试完整模型管线")

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(" 钢琴演奏分析模型测试流程 ".center(60, "="))
    print("=" * 60)
    
    args = parse_args()
    
    # 1. 修复编码问题
    fix_encoding()
    
    # 2. 下载模型（如果需要）
    if not args.skip_download:
        download_model(args.model)
    
    # 3. 运行测试
    if args.test_audiomae or not (args.test_pipeline or args.test_qformer):
        test_audiomae()
    
    if args.test_qformer or not (args.test_pipeline or args.test_audiomae):
        test_qformer()
    
    if args.test_pipeline or not (args.test_audiomae or args.test_qformer):
        test_pipeline()
    
    print("\n" + "=" * 60)
    print(" 测试流程完成 ".center(60, "="))
    print("=" * 60)

if __name__ == "__main__":
    main() 