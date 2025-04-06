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
        default="Qwen/QwQ-32B",  # 使用QwQ-32B模型
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

def run_command(command):
    """运行命令并打印输出"""
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

def download_model_with_cli(model_name, revision="main", cache_dir=None):
    """使用huggingface-cli下载模型"""
    logger.info(f"开始下载模型: {model_name}")
    
    # 构建命令
    cmd = f"huggingface-cli download {model_name} --revision {revision}"
    
    # 如果指定了缓存目录
    if cache_dir:
        cache_dir = os.path.abspath(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        cmd += f" --cache-dir {cache_dir}"
    
    # 执行下载命令
    logger.info(f"执行命令: {cmd}")
    return_code = run_command(cmd)
    
    if return_code == 0:
        logger.info(f"模型 {model_name} 下载成功")
        return True
    else:
        logger.error(f"模型 {model_name} 下载失败，返回码: {return_code}")
        return False

def update_config_file(model_name):
    """更新配置文件中的模型路径"""
    config_path = "config/audio_config.yaml"
    
    try:
        # 确保配置目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # 读取配置文件（如果存在）
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            # 创建默认配置
            config = {
                'audio_model': {
                    'model_type': 'audiomae',
                    'pretrained_path': 'pretrained/audiomae_base.pth',
                    'sample_rate': 32000,
                    'n_mels': 128,
                    'feature_dim': 1024,
                    'max_length': 4096,
                    'frameshift': 10
                },
                'qformer': {
                    'num_query_tokens': 32,
                    'cross_attention_freq': 2
                }
            }
        
        # 更新或添加LLM配置
        if 'llm' not in config:
            config['llm'] = {}
            
        config['llm']['model_name_or_path'] = model_name
        config['llm']['model_type'] = 'auto'
        
        # 写入配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
            
        logger.info(f"配置文件 {config_path} 已更新，使用模型: {model_name}")
        return True
    except Exception as e:
        logger.error(f"更新配置文件失败: {e}")
        return False

def main():
    """主函数"""
    print("\n" + "=" * 60)
    print(" LLM模型下载工具 ".center(60, "="))
    print("=" * 60)
    
    args = parse_args()
    
    # 下载模型
    success = download_model_with_cli(
        model_name=args.model,
        revision=args.revision,
        cache_dir=args.cache_dir
    )
    
    if success:
        # 更新配置文件
        update_config_file(args.model)
        print("\n" + "=" * 60)
        print(" 模型下载完成，现在可以运行测试 ".center(60, "="))
        print("=" * 60)
        print("\n运行测试命令:")
        print("python test_model_pipeline.py")
    else:
        print("\n" + "=" * 60)
        print(" 模型下载失败 ".center(60, "="))
        print("=" * 60)
        print("\n您可以尝试:")
        print("1. 检查网络连接")
        print("2. 确认huggingface-cli配置正确")
        print("3. 尝试下载较小的模型，例如: --model Qwen/QwQ-7B")
        sys.exit(1)

if __name__ == "__main__":
    main() 