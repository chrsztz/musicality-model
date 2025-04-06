import os
import torch
import logging
import yaml
import numpy as np
from typing import Dict, List, Optional
from torch.utils.data import Dataset
from datasets import load_dataset
import sys

# 修复导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.preprocessing.audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuroPianoHFDataset(Dataset):
    """
    使用HuggingFace datasets加载的NeuroPiano钢琴演奏数据集
    包含104条录音，3.3k问答对，12个演奏维度的评估
    
    Dataset结构:
    - audio_path: 采样率为48000的加载音频数组
    - piece: 练习的名称
    - subject: 评估者ID
    - question: 日语问题
    - answer: 日语回答
    - q_eng: 英语问题
    - a_eng: 英语回答
    - score: 以6分制评分的数值
    """
    
    def __init__(
        self, 
        config_path: str = "config/audio_config.yaml",
        split: str = "train",
        transform: Optional[callable] = None,
        max_qa_pairs: Optional[int] = None,
        use_english: bool = True,
    ):
        """
        初始化NeuroPiano HuggingFace数据集
        
        Args:
            config_path: 配置文件路径
            split: 数据集分割 ('train', 'validation', 'test')
            transform: 数据转换函数
            max_qa_pairs: 最大问答对数量 (用于调试)
            use_english: 是否使用英文版本的问答对
        """
        self.split = split
        self.transform = transform
        self.max_qa_pairs = max_qa_pairs
        self.use_english = use_english
        
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded configuration from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, creating default config")
            # 如果配置文件不存在，创建默认配置
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            config = {
                'audio_model': {
                    'sample_rate': 32000,
                    'n_mels': 128,
                    'max_length': 4096,
                    'frameshift': 10
                },
                'datasets': {
                    'neuropiano': {
                        'train_split': 0.8,
                        'val_split': 0.1,
                        'test_split': 0.1
                    }
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
        
        # 初始化音频处理器
        self.audio_processor = AudioProcessor(config_path)
        
        # 加载HuggingFace数据集
        try:
            self.dataset = load_dataset("anusfoil/NeuroPiano-data", split=split)
            logger.info(f"Successfully loaded NeuroPiano dataset from HuggingFace, split: {split}")
            
            # 如果设置了最大问答对数量，进行截断
            if self.max_qa_pairs is not None and len(self.dataset) > self.max_qa_pairs:
                self.dataset = self.dataset.select(range(self.max_qa_pairs))
                
            logger.info(f"Dataset contains {len(self.dataset)} entries")
        except Exception as e:
            logger.error(f"Error loading dataset from HuggingFace: {e}")
            raise
    
    def __len__(self) -> int:
        """返回数据集长度"""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取数据集项
        
        Args:
            idx: 索引
            
        Returns:
            包含音频特征和问答对的字典
        """
        item = self.dataset[idx]
        
        # 确定使用哪种语言的问答对
        question = item['q_eng'] if self.use_english else item['question']
        answer = item['a_eng'] if self.use_english else item['answer']
        
        try:
            # 处理音频数据
            # 注意：HF数据集中的audio_path可能是一个特殊结构，需要根据实际情况调整
            # 这里假设audio_path是一个numpy数组
            audio_data = item['audio_path']
            
            # 检查音频数据类型并进行相应处理
            if isinstance(audio_data, np.ndarray):
                # 直接转换numpy数组到torch张量
                audio_tensor = torch.tensor(audio_data).float().unsqueeze(0)
            elif isinstance(audio_data, dict) and 'array' in audio_data:
                # 如果是HF的Audio结构，可能会是包含'array'键的字典
                audio_tensor = torch.tensor(audio_data['array']).float().unsqueeze(0)
            else:
                # 未知类型，记录错误并返回占位数据
                logger.warning(f"Unknown audio data type: {type(audio_data)}")
                return {
                    "piece": item["piece"],
                    "subject": item["subject"],
                    "mel_spectrogram": torch.zeros(1, self.audio_processor.n_mels, self.audio_processor.max_length),
                    "question": question,
                    "answer": answer,
                    "score": torch.tensor(item["score"], dtype=torch.float) if "score" in item else None,
                    "error": f"Unknown audio data type: {type(audio_data)}"
                }
                
            # 如果是立体声，转为单声道
            if audio_tensor.shape[0] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
            
            # 处理音频特征
            mel_spec = self.audio_processor.extract_mel_spectrogram(audio_tensor)
            normalized_spec = self.audio_processor.normalize_spectrogram(mel_spec)
            padded_spec = self.audio_processor.pad_or_truncate(normalized_spec)
            
            # 应用变换（如果有）
            if self.transform is not None:
                padded_spec = self.transform(padded_spec)
            
            # 构建返回项
            result = {
                "piece": item["piece"],
                "subject": item["subject"],
                "mel_spectrogram": padded_spec,
                "question": question,
                "answer": answer,
                "score": torch.tensor(item["score"], dtype=torch.float) if "score" in item else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            # 返回一个占位项
            return {
                "piece": item["piece"],
                "subject": item["subject"],
                "mel_spectrogram": torch.zeros(1, self.audio_processor.n_mels, self.audio_processor.max_length),
                "question": question,
                "answer": answer,
                "score": torch.tensor(item["score"], dtype=torch.float) if "score" in item else None,
                "error": str(e)
            }

    def get_all_questions(self) -> List[str]:
        """
        获取数据集中的所有问题
        
        Returns:
            问题列表
        """
        question_field = 'q_eng' if self.use_english else 'question'
        return self.dataset[question_field]
    
    def get_question_types(self) -> Dict[str, int]:
        """
        统计问题类型及其频率
        
        Returns:
            问题类型统计字典
        """
        question_field = 'q_eng' if self.use_english else 'question'
        questions = self.dataset[question_field]
        
        question_types = {}
        for q in questions:
            # 简单处理：提取问题的前几个词作为类型
            q_type = " ".join(q.split()[:3]) + "..."
            question_types[q_type] = question_types.get(q_type, 0) + 1
        
        return dict(sorted(question_types.items(), key=lambda x: x[1], reverse=True))


# 测试代码
if __name__ == "__main__":
    # 创建配置文件目录（如果不存在）
    os.makedirs("config", exist_ok=True)
    
    dataset = NeuroPianoHFDataset(
        split="train",
        max_qa_pairs=10  # 仅用于测试
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # 测试获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample piece: {sample['piece']}")
        print(f"Sample subject: {sample['subject']}")
        print(f"Sample mel_spectrogram shape: {sample['mel_spectrogram'].shape}")
        print(f"Sample question: {sample['question']}")
        print(f"Sample answer: {sample['answer']}")
        print(f"Sample score: {sample['score']}")
        
        # 查看问题类型
        question_types = dataset.get_question_types()
        print("\nQuestion types:")
        for q_type, count in list(question_types.items())[:5]:  # 只显示前5种
            print(f"  - {q_type}: {count}")
    else:
        print("Dataset is empty")