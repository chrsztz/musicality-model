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
        default="facebook/opt-125m",  # 使用公开可访问的小型模型
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

# 其余部分保持不变 