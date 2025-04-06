import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Union
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ver3.data.preprocessing.audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuroPianoDataset(Dataset):
    """
    NeuroPiano钢琴演奏数据集
    包含104条录音，3.3k问答对，12个演奏维度的评估
    """
    
    def __init__(
        self, 
        data_path: str,
        config_path: str = "config/audio_config.yaml",
        split: str = "train",
        transform: Optional[callable] = None,
        max_qa_pairs: Optional[int] = None,
    ):
        """
        初始化NeuroPiano数据集
        
        Args:
            data_path: 数据集根目录路径
            config_path: 配置文件路径
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据转换函数
            max_qa_pairs: 最大问答对数量 (用于调试)
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.max_qa_pairs = max_qa_pairs
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 初始化音频处理器
        self.audio_processor = AudioProcessor(config_path)
        
        # 加载数据集分割信息
        split_ratio = config['datasets']['neuropiano']
        self.train_ratio = split_ratio['train_split']
        self.val_ratio = split_ratio['val_split']
        self.test_ratio = split_ratio['test_split']
        
        # 加载数据集
        self.audio_files, self.qa_pairs = self._load_dataset()
        logger.info(f"Loaded NeuroPiano {split} split with {len(self.qa_pairs)} QA pairs from {len(self.audio_files)} audio files")
    
    def _load_dataset(self) -> Tuple[List[str], List[Dict]]:
        """
        加载数据集文件和问答对
        
        Returns:
            音频文件列表和问答对列表
        """
        # 检查数据路径是否存在
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        
        # 加载音频文件
        audio_dir = os.path.join(self.data_path, "audio")
        if not os.path.exists(audio_dir):
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
        
        all_audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) 
                          if f.endswith(('.wav', '.mp3', '.flac'))]
        all_audio_files.sort()  # 确保顺序一致
        
        # 加载注释文件
        annotations_path = os.path.join(self.data_path, "annotations.json")
        if not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        # 根据分割比例分配数据
        n_files = len(all_audio_files)
        train_end = int(n_files * self.train_ratio)
        val_end = train_end + int(n_files * self.val_ratio)
        
        if self.split == "train":
            audio_files = all_audio_files[:train_end]
            audio_ids = [os.path.basename(f).split('.')[0] for f in audio_files]
        elif self.split == "val":
            audio_files = all_audio_files[train_end:val_end]
            audio_ids = [os.path.basename(f).split('.')[0] for f in audio_files]
        elif self.split == "test":
            audio_files = all_audio_files[val_end:]
            audio_ids = [os.path.basename(f).split('.')[0] for f in audio_files]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # 提取相应的问答对
        qa_pairs = []
        for audio_id in audio_ids:
            if audio_id in annotations:
                for qa in annotations[audio_id]:
                    qa_entry = {
                        "audio_id": audio_id,
                        "audio_path": os.path.join(audio_dir, f"{audio_id}.wav"),
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "rating": qa.get("rating", None)  # 如果有评分
                    }
                    qa_pairs.append(qa_entry)
        
        # 如果设置了最大问答对数量，进行截断
        if self.max_qa_pairs is not None and len(qa_pairs) > self.max_qa_pairs:
            qa_pairs = qa_pairs[:self.max_qa_pairs]
        
        return audio_files, qa_pairs
    
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.qa_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取数据集项
        
        Args:
            idx: 索引
            
        Returns:
            包含音频特征和问答对的字典
        """
        qa_pair = self.qa_pairs[idx]
        audio_path = qa_pair["audio_path"]
        
        # 处理音频特征
        try:
            audio_features = self.audio_processor.process_audio(audio_path)
            
            # 应用变换（如果有）
            if self.transform is not None:
                audio_features = self.transform(audio_features)
            
            # 合并特征和问答信息
            item = {
                "audio_id": qa_pair["audio_id"],
                "mel_spectrogram": audio_features["mel_spectrogram"],
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
            }
            
            # 添加评分（如果有）
            if qa_pair["rating"] is not None:
                item["rating"] = torch.tensor(qa_pair["rating"], dtype=torch.float)
            
            return item
            
        except Exception as e:
            logger.error(f"Error processing item {idx} (audio: {audio_path}): {e}")
            # 返回一个占位项
            return {
                "audio_id": qa_pair["audio_id"],
                "mel_spectrogram": torch.zeros(1, self.audio_processor.n_mels, self.audio_processor.max_length),
                "question": qa_pair["question"],
                "answer": qa_pair["answer"],
                "error": str(e)
            }

    def get_all_questions(self) -> List[str]:
        """
        获取数据集中的所有问题
        
        Returns:
            问题列表
        """
        return [qa["question"] for qa in self.qa_pairs]
    
    def get_question_types(self) -> Dict[str, int]:
        """
        统计问题类型及其频率
        
        Returns:
            问题类型统计字典
        """
        question_types = {}
        for qa in self.qa_pairs:
            q = qa["question"]
            # 简单处理：提取问题的前几个词作为类型
            q_type = " ".join(q.split()[:3]) + "..."
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return dict(sorted(question_types.items(), key=lambda x: x[1], reverse=True))


# 测试代码
if __name__ == "__main__":
    # 此处为测试代码，实际使用时需提供正确的数据路径
    test_data_path = "data/neuropiano"
    
    if os.path.exists(test_data_path):
        dataset = NeuroPianoDataset(
            data_path=test_data_path,
            split="train",
            max_qa_pairs=10  # 仅用于测试
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # 测试获取一个样本
        sample = dataset[0]
        print(f"Sample audio_id: {sample['audio_id']}")
        print(f"Sample mel_spectrogram shape: {sample['mel_spectrogram'].shape}")
        print(f"Sample question: {sample['question']}")
        print(f"Sample answer: {sample['answer']}")
        
        # 查看问题类型
        question_types = dataset.get_question_types()
        print("\nQuestion types:")
        for q_type, count in list(question_types.items())[:5]:  # 只显示前5种
            print(f"  - {q_type}: {count}")
    else:
        print(f"Test data path not found: {test_data_path}")
        print("Please setup the correct data path for testing.")