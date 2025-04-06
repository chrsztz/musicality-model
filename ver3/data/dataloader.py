import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets.neuropiano_hf import NeuroPianoHFDataset
# 后续可以导入其他数据集模块
# from data.datasets.expert_novice import ExpertNoviceDataset
# from data.datasets.crocus import CrocusPianoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PianoDataLoader:
    """用于钢琴演奏数据的数据加载器"""
    
    def __init__(
        self, 
        config_path: str = "config/audio_config.yaml",
        use_neuropiano: bool = True,
        use_expert_novice: bool = False,
        use_crocus: bool = False,
        batch_size: Optional[int] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        """
        初始化数据加载器
        
        Args:
            config_path: 配置文件路径
            use_neuropiano: 是否使用NeuroPiano数据集
            use_expert_novice: 是否使用Expert-Novice数据集
            use_crocus: 是否使用CROCUS数据集
            batch_size: 批大小 (若None则使用配置文件中的值)
            num_workers: 数据加载线程数
            pin_memory: 是否使用内存锁定（用于GPU训练）
        """
        self.config_path = config_path
        
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, creating default config")
            # 如果配置文件不存在，创建默认配置
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            self.config = {
                'training': {
                    'batch_size': 8
                },
                'datasets': {
                    'neuropiano': {
                        'path': 'data/neuropiano',
                        'train_split': 0.8,
                        'val_split': 0.1,
                        'test_split': 0.1
                    }
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f)
        
        # 设置批大小
        self.batch_size = batch_size if batch_size is not None else self.config['training']['batch_size']
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 数据集选择标志
        self.use_neuropiano = use_neuropiano
        self.use_expert_novice = use_expert_novice
        self.use_crocus = use_crocus
        
        # 数据集路径
        self.dataset_paths = {
            'neuropiano': self.config['datasets']['neuropiano']['path'],
            'expert_novice': self.config['datasets']['expert_novice']['path'] if 'expert_novice' in self.config['datasets'] else None,
            'crocus': self.config['datasets']['crocus']['path'] if 'crocus' in self.config['datasets'] else None,
        }
        
        logger.info(f"Initialized PianoDataLoader with batch_size={self.batch_size}, num_workers={num_workers}")
    
    def _create_dataset(self, dataset_type: str, split: str) -> Optional[torch.utils.data.Dataset]:
        """
        创建指定类型和分割的数据集
        
        Args:
            dataset_type: 数据集类型
            split: 数据集分割
            
        Returns:
            数据集对象或None
        """
        if dataset_type == 'neuropiano' and self.use_neuropiano:
            # 使用HuggingFace版本的NeuroPiano数据集
            # 转换split名称，将'val'和'test'转换为'eval'以匹配Hugging Face数据集的命名
            hf_split = 'eval' if split in ['val', 'test'] else split
            
            try:
                return NeuroPianoHFDataset(
                    config_path=self.config_path,
                    split=hf_split,
                    use_english=True  # 默认使用英文问答对
                )
            except Exception as e:
                logger.error(f"Failed to load NeuroPiano dataset: {e}")
                return None
        
        elif dataset_type == 'expert_novice' and self.use_expert_novice:
            # 此处添加Expert-Novice数据集的加载代码
            # 当前版本中暂未实现
            logger.info("Expert-Novice dataset loading not implemented yet")
            return None
            
        elif dataset_type == 'crocus' and self.use_crocus:
            # 此处添加CROCUS数据集的加载代码
            # 当前版本中暂未实现
            logger.info("CROCUS dataset loading not implemented yet")
            return None
            
        return None
    
    def _create_combined_dataset(self, split: str) -> torch.utils.data.Dataset:
        """
        创建组合数据集
        
        Args:
            split: 数据集分割
            
        Returns:
            组合数据集
        """
        datasets = []
        
        # 尝试加载各个数据集
        for dataset_type in ['neuropiano', 'expert_novice', 'crocus']:
            dataset = self._create_dataset(dataset_type, split)
            if dataset is not None:
                datasets.append(dataset)
                logger.info(f"Added {dataset_type} {split} dataset with {len(dataset)} samples")
        
        if not datasets:
            raise ValueError("No datasets available. Please check dataset paths and flags.")
        
        # 如果只有一个数据集，直接返回
        if len(datasets) == 1:
            return datasets[0]
        
        # 否则，使用ConcatDataset组合多个数据集
        return ConcatDataset(datasets)
    
    def get_dataloader(self, split: str) -> torch.utils.data.DataLoader:
        """
        获取指定分割的数据加载器
        
        Args:
            split: 数据集分割 ('train', 'val', 'test')
            
        Returns:
            数据加载器
        """
        dataset = self._create_combined_dataset(split)
        
        # 为训练集创建数据加载器
        if split == 'train':
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True
            )
        # 为验证集和测试集创建数据加载器
        else:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
    
    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """获取训练集数据加载器"""
        return self.get_dataloader('train')
    
    def get_val_dataloader(self) -> torch.utils.data.DataLoader:
        """获取验证集数据加载器"""
        return self.get_dataloader('val')
    
    def get_test_dataloader(self) -> torch.utils.data.DataLoader:
        """获取测试集数据加载器"""
        return self.get_dataloader('test')
    
    def get_all_dataloaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        """
        获取所有数据集分割的数据加载器
        
        Returns:
            包含三个分割数据加载器的字典
        """
        return {
            'train': self.get_train_dataloader(),
            'val': self.get_val_dataloader(),
            'test': self.get_test_dataloader()
        }


# 测试代码
if __name__ == "__main__":
    # 此处为测试代码，实际使用时需提供正确的配置和数据路径
    
    # 创建配置文件目录（如果不存在）
    os.makedirs("config", exist_ok=True)
    
    # 如果配置文件不存在，创建一个临时配置
    if not os.path.exists("config/audio_config.yaml"):
        with open("config/audio_config.yaml", "w") as f:
            f.write("""
training:
  batch_size: 4
  
datasets:
  neuropiano:
    path: data/neuropiano
    train_split: 0.8
    val_split: 0.1
    test_split: 0.1
            """)
    
    # 数据加载器测试
    try:
        dataloader = PianoDataLoader(
            batch_size=2,
            num_workers=0  # 调试时使用单线程
        )
        
        # 检查数据集路径
        if os.path.exists(dataloader.dataset_paths['neuropiano']):
            # 获取数据加载器
            train_loader = dataloader.get_train_dataloader()
            
            print(f"Train dataloader created with {len(train_loader)} batches")
            
            # 测试读取一个批次
            for batch_idx, batch in enumerate(train_loader):
                print(f"Batch {batch_idx}:")
                print(f"  - Spectrogram shape: {batch['mel_spectrogram'].shape}")
                print(f"  - Question: {batch['question'][0]}")
                print(f"  - Answer: {batch['answer'][0][:50]}...")  # 只显示前50个字符
                
                # 只测试一个批次
                break
        else:
            print(f"NeuroPiano dataset path not found: {dataloader.dataset_paths['neuropiano']}")
            print("Please setup the correct data path for testing.")
    except Exception as e:
        print(f"Error testing dataloader: {e}")