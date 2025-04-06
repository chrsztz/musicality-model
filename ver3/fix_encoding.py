"""
修复项目中的编码问题

这个脚本会遍历所有Python和YAML文件，确保它们在读取和写入时使用UTF-8编码
"""

import os
import sys
import yaml
import glob
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def fix_yaml_file(file_path):
    """修复YAML文件的编码问题"""
    try:
        # 尝试以UTF-8读取
        with open(file_path, 'r', encoding='utf-8') as f:
            content = yaml.safe_load(f)
        
        # 写回文件，确保使用UTF-8编码
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(content, f, allow_unicode=True)
        
        logger.info(f"✓ 已修复YAML文件: {file_path}")
        return True
    except Exception as e:
        # 如果UTF-8读取失败，尝试用其他编码
        try:
            # 尝试用系统默认编码读取
            with open(file_path, 'r') as f:
                content = yaml.safe_load(f)
            
            # 写回文件，确保使用UTF-8编码
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(content, f, allow_unicode=True)
            
            logger.info(f"✓ 已修复YAML文件(使用系统默认编码): {file_path}")
            return True
        except Exception as inner_e:
            logger.error(f"× 无法修复YAML文件: {file_path}, 错误: {inner_e}")
            return False

def fix_python_file_encoding(file_path):
    """修复Python文件的编码问题，更新所有文件读写操作确保使用UTF-8编码"""
    try:
        # 尝试以UTF-8读取
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查并修改文件打开语句
        modified = False
        
        # 替换不带编码的open调用
        if "open(" in content and "encoding=" not in content:
            # 替换: open(file, 'r') -> open(file, 'r', encoding='utf-8')
            content = content.replace("open(", "open(").replace(", 'r')", ", 'r', encoding='utf-8')")
            content = content.replace(", 'w')", ", 'w', encoding='utf-8')")
            content = content.replace(", \"r\")", ", \"r\", encoding=\"utf-8\")")
            content = content.replace(", \"w\")", ", \"w\", encoding=\"utf-8\")")
            modified = True
        
        # 写回文件
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✓ 已更新Python文件中的编码: {file_path}")
        else:
            logger.info(f"✓ Python文件无需修改: {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"× 处理Python文件时出错: {file_path}, 错误: {e}")
        return False

def fix_all_files(directory):
    """修复目录下所有文件的编码问题"""
    # 统计信息
    total_files = 0
    fixed_files = 0
    failed_files = 0
    
    # 处理YAML文件
    yaml_files = glob.glob(f"{directory}/**/*.yaml", recursive=True)
    yaml_files += glob.glob(f"{directory}/**/*.yml", recursive=True)
    
    for yaml_file in yaml_files:
        total_files += 1
        if fix_yaml_file(yaml_file):
            fixed_files += 1
        else:
            failed_files += 1
    
    # 处理Python文件
    python_files = glob.glob(f"{directory}/**/*.py", recursive=True)
    
    for py_file in python_files:
        # 跳过当前脚本
        if os.path.samefile(py_file, __file__):
            continue
        
        total_files += 1
        if fix_python_file_encoding(py_file):
            fixed_files += 1
        else:
            failed_files += 1
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info(f"处理完成！共处理 {total_files} 个文件")
    logger.info(f"成功修复: {fixed_files} 个文件")
    logger.info(f"修复失败: {failed_files} 个文件")
    logger.info("=" * 50)

def create_config_directory():
    """确保配置目录存在"""
    os.makedirs("config", exist_ok=True)
    
    # 检查是否存在音频配置文件，如果不存在则创建
    config_path = "config/audio_config.yaml"
    if not os.path.exists(config_path):
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
            },
            'qformer': {
                'num_query_tokens': 32,
                'cross_attention_freq': 2
            },
            'llm': {
                'model_type': 'chatglm',
                'model_name_or_path': 'THUDM/chatglm3-6b',
                'projection_dim': 4096
            }
        }
        
        # 写入配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        
        logger.info(f"✓ 已创建默认配置文件: {config_path}")

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" 编码问题修复工具 ".center(50, "="))
    print("=" * 50)
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建配置目录
    create_config_directory()
    
    # 修复所有文件
    print(f"\n开始处理目录: {base_dir}\n")
    fix_all_files(base_dir)
    
    print("\n编码修复完成！请重新运行测试脚本以验证修复效果。")
    print("=" * 50) 