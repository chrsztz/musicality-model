"""
下载测试所需的LLM模型

使用huggingface-cli下载模型，确保测试前已准备好模型
"""

import os
import sys
import subprocess
import logging
import yaml
import argparse
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='下载测试所需的LLM模型')
    parser.add_argument(
        '--model', 
        type=str, 
        default="facebook/opt-125m",  # 使用公开可访问的小型模型
        help='要下载的模型名称'
    )
    parser.add_argument(
        '--revision', 
        type=str, 
        default="main",
        help='模型分支或版本'
    )
    parser.add_argument(
        '--cache_dir', 
        type=str, 
        default=None,
        help='下载模型的缓存目录'
    )
    return parser.parse_args()

# 其余部分保持不变 