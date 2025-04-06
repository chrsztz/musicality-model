import os
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datasets import load_dataset
import nltk
import sys
from collections import Counter

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 尝试下载NLTK资源（如果需要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class NeuroPianoDataProcessor:
    """
    处理NeuroPiano数据集，提取和组织特征
    包括从问答对中提取评价、技术分析、音乐描述等
    """
    
    def __init__(
        self, 
        use_english: bool = True,
        cache_features: bool = True,
        cache_dir: str = "data/cache"
    ):
        """
        初始化NeuroPiano数据处理器
        
        Args:
            use_english: 是否使用英文版本的数据
            cache_features: 是否缓存提取的特征
            cache_dir: 特征缓存目录
        """
        self.use_english = use_english
        self.cache_features = cache_features
        self.cache_dir = cache_dir
        
        # 确保缓存目录存在
        if self.cache_features:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 加载NeuroPiano数据集
        try:
            logger.info("Loading NeuroPiano dataset...")
            self.dataset = load_dataset("anusfoil/NeuroPiano-data")
            logger.info(f"Successfully loaded NeuroPiano dataset with splits: {self.dataset.keys()}")
        except Exception as e:
            logger.error(f"Error loading NeuroPiano dataset: {e}")
            self.dataset = None
            
        # 存储提取的特征
        self.features = {
            'good_or_bad': {},  # 好/坏评价
            'questions': {},    # 问题类型
            'scores': {},       # 分数
            'techniques': {},   # 技术分析
            'attributes': {},   # 属性（形容词）
            'performance_dimensions': {}  # 演奏维度
        }
    
    def extract_all_features(self) -> Dict[str, Any]:
        """
        提取所有NeuroPiano数据集特征
        
        Returns:
            提取的特征字典
        """
        if self.dataset is None:
            logger.error("Cannot extract features: dataset not loaded")
            return self.features
        
        # 检查缓存
        cache_path = os.path.join(self.cache_dir, f"neuropiano_features_{'en' if self.use_english else 'jp'}.pt")
        if self.cache_features and os.path.exists(cache_path):
            logger.info(f"Loading cached features from {cache_path}")
            try:
                self.features = torch.load(cache_path)
                return self.features
            except Exception as e:
                logger.warning(f"Failed to load cached features: {e}")
        
        # 提取各种特征
        logger.info("Extracting NeuroPiano features...")
        
        for split in self.dataset.keys():
            logger.info(f"Processing {split} split...")
            
            # 获取问答字段
            q_field = 'q_eng' if self.use_english else 'question'
            a_field = 'a_eng' if self.use_english else 'answer'
            
            # 迭代数据集中的每一项
            for item in self.dataset[split]:
                # 提取问题和答案
                question = item[q_field]
                answer = item[a_field]
                piece = item['piece']
                
                # 提取各类特征
                self._extract_good_or_bad(answer, piece)
                self._extract_question_type(question, piece)
                self._extract_score(item, piece)
                self._extract_techniques(answer, piece)
                self._extract_attributes(answer, piece)
        
        # 处理和汇总提取的特征
        self._process_extracted_features()
        
        # 缓存特征
        if self.cache_features:
            logger.info(f"Caching features to {cache_path}")
            try:
                torch.save(self.features, cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache features: {e}")
        
        return self.features
    
    def _extract_good_or_bad(self, answer: str, piece: str) -> None:
        """提取好/坏评价"""
        # 简单的好/坏评价提取
        positive_words = ['good', 'great', 'excellent', 'impressive', 'beautiful', 'well']
        negative_words = ['bad', 'poor', 'weak', 'lacking', 'inconsistent', 'needs improvement']
        
        # 对答案进行分词
        tokens = nltk.word_tokenize(answer.lower())
        
        # 计算正面和负面词的数量
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        # 判断总体评价
        if positive_count > negative_count:
            sentiment = 'positive'
        elif negative_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # 存储结果
        if piece not in self.features['good_or_bad']:
            self.features['good_or_bad'][piece] = []
        
        self.features['good_or_bad'][piece].append({
            'sentiment': sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count
        })
    
    def _extract_question_type(self, question: str, piece: str) -> None:
        """提取问题类型"""
        # 分类问题类型
        if 'difficulty' in question.lower() or 'challenging' in question.lower():
            q_type = 'difficulty'
        elif 'emotional' in question.lower() or 'expression' in question.lower():
            q_type = 'emotion'
        elif 'technique' in question.lower() or 'technical' in question.lower():
            q_type = 'technique'
        elif 'interpretati' in question.lower():
            q_type = 'interpretation'
        elif 'style' in question.lower():
            q_type = 'style'
        else:
            q_type = 'other'
        
        # 存储结果
        if piece not in self.features['questions']:
            self.features['questions'][piece] = []
        
        self.features['questions'][piece].append({
            'question': question,
            'type': q_type
        })
    
    def _extract_score(self, item: Dict, piece: str) -> None:
        """提取分数"""
        if 'score' in item and item['score'] is not None:
            # 存储结果
            if piece not in self.features['scores']:
                self.features['scores'][piece] = []
            
            self.features['scores'][piece].append({
                'score': item['score']
            })
    
    def _extract_techniques(self, answer: str, piece: str) -> None:
        """提取技术分析"""
        # 常见的钢琴技术术语
        technique_terms = [
            'pedal', 'legato', 'staccato', 'articulation', 'fingering', 
            'dynamics', 'touch', 'phrasing', 'voicing', 'balance',
            'tempo', 'rhythm', 'rubato', 'timing', 'control'
        ]
        
        # 检查答案中包含哪些技术术语
        found_techniques = []
        for term in technique_terms:
            if term in answer.lower():
                found_techniques.append(term)
        
        # 存储结果
        if piece not in self.features['techniques']:
            self.features['techniques'][piece] = []
        
        if found_techniques:
            self.features['techniques'][piece].append({
                'techniques': found_techniques
            })
    
    def _extract_attributes(self, answer: str, piece: str) -> None:
        """提取属性（形容词）"""
        # 使用NLTK提取形容词（简化版）
        tokens = nltk.word_tokenize(answer.lower())
        
        # 常见的音乐相关形容词
        music_adjectives = [
            'beautiful', 'elegant', 'precise', 'emotional', 'expressive',
            'technical', 'controlled', 'balanced', 'dynamic', 'sensitive',
            'nuanced', 'dramatic', 'powerful', 'delicate', 'refined',
            'passionate', 'calm', 'energetic', 'bold', 'subtle',
            'clear', 'confused', 'consistent', 'inconsistent'
        ]
        
        # 找到答案中包含的音乐形容词
        found_adjectives = [word for word in tokens if word in music_adjectives]
        
        # 存储结果
        if piece not in self.features['attributes']:
            self.features['attributes'][piece] = []
        
        if found_adjectives:
            self.features['attributes'][piece].append({
                'adjectives': found_adjectives
            })
    
    def _process_extracted_features(self) -> None:
        """处理和汇总提取的特征"""
        # 处理好/坏评价
        for piece, sentiments in self.features['good_or_bad'].items():
            sentiment_counts = Counter([s['sentiment'] for s in sentiments])
            dominant_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else 'neutral'
            self.features['good_or_bad'][piece] = {
                'dominant_sentiment': dominant_sentiment,
                'sentiment_distribution': dict(sentiment_counts)
            }
        
        # 处理问题类型
        for piece, questions in self.features['questions'].items():
            question_types = Counter([q['type'] for q in questions])
            self.features['questions'][piece] = {
                'question_types': dict(question_types),
                'total_questions': len(questions)
            }
        
        # 处理分数
        for piece, scores in self.features['scores'].items():
            score_values = [s['score'] for s in scores]
            self.features['scores'][piece] = {
                'average_score': sum(score_values) / len(score_values) if score_values else 0,
                'min_score': min(score_values) if score_values else 0,
                'max_score': max(score_values) if score_values else 0,
                'score_count': len(score_values)
            }
        
        # 处理技术分析
        for piece, techniques in self.features['techniques'].items():
            all_techniques = []
            for t in techniques:
                all_techniques.extend(t['techniques'])
            
            technique_counts = Counter(all_techniques)
            self.features['techniques'][piece] = {
                'technique_counts': dict(technique_counts),
                'total_techniques': len(all_techniques)
            }
        
        # 处理属性
        for piece, attributes in self.features['attributes'].items():
            all_adjectives = []
            for a in attributes:
                all_adjectives.extend(a['adjectives'])
            
            adjective_counts = Counter(all_adjectives)
            self.features['attributes'][piece] = {
                'adjective_counts': dict(adjective_counts),
                'total_adjectives': len(all_adjectives)
            }
    
    def get_piece_features(self, piece: str) -> Dict[str, Any]:
        """
        获取特定乐曲的所有特征
        
        Args:
            piece: 乐曲名称
            
        Returns:
            乐曲特征字典
        """
        result = {}
        for feature_type, feature_data in self.features.items():
            if piece in feature_data:
                result[feature_type] = feature_data[piece]
        
        return result
    
    def get_performance_metrics(self, piece: str) -> Dict[str, float]:
        """
        获取乐曲的演奏指标
        
        Args:
            piece: 乐曲名称
            
        Returns:
            演奏指标字典
        """
        # 基于前面提取的特征计算演奏指标
        metrics = {}
        
        # 从分数计算整体质量
        if piece in self.features['scores']:
            metrics['overall_quality'] = self.features['scores'][piece]['average_score'] / 6.0  # 归一化到0-1
        
        # 从技术分析计算技术指标
        if piece in self.features['techniques']:
            technique_counts = self.features['techniques'][piece]['technique_counts']
            total = sum(technique_counts.values())
            
            # 计算各个技术维度的比例
            for technique, count in technique_counts.items():
                metrics[f'technique_{technique}'] = count / total if total > 0 else 0
        
        # 从情感分析计算情感指标
        if piece in self.features['good_or_bad']:
            sentiment = self.features['good_or_bad'][piece]['dominant_sentiment']
            metrics['sentiment'] = {'positive': 1.0, 'neutral': 0.5, 'negative': 0.0}[sentiment]
        
        return metrics
    
    def enrich_feature_embeddings(self, 
                                 embeddings: torch.Tensor, 
                                 piece: str) -> torch.Tensor:
        """
        用NeuroPiano特征丰富音频嵌入
        
        Args:
            embeddings: 原始音频特征嵌入 [B, N, D]
            piece: 乐曲名称
            
        Returns:
            增强后的嵌入 [B, N, D]
        """
        # 获取特征
        metrics = self.get_performance_metrics(piece)
        
        if not metrics:
            return embeddings  # 如果没有找到特征，返回原始嵌入
        
        # 创建特征向量
        feature_values = list(metrics.values())
        feature_tensor = torch.tensor(feature_values, dtype=torch.float32)
        
        # 标准化
        if len(feature_tensor) > 0:
            feature_tensor = (feature_tensor - feature_tensor.mean()) / (feature_tensor.std() + 1e-8)
            
            # 重塑为 [1, 1, F]
            feature_tensor = feature_tensor.reshape(1, 1, -1)
            
            # 调整大小以匹配嵌入维度
            B, N, D = embeddings.shape
            if feature_tensor.shape[-1] < D:
                # 如果特征维度小于嵌入维度，填充为0
                padding = torch.zeros(1, 1, D - feature_tensor.shape[-1], device=embeddings.device)
                feature_tensor = torch.cat([feature_tensor, padding], dim=-1)
            elif feature_tensor.shape[-1] > D:
                # 如果特征维度大于嵌入维度，截断
                feature_tensor = feature_tensor[:, :, :D]
            
            # 扩展到相同批次和序列长度 [B, N, D]
            feature_tensor = feature_tensor.expand(B, N, -1)
            
            # 将特征添加到嵌入
            enhanced_embeddings = embeddings + 0.1 * feature_tensor  # 添加缩放后的特征
            
            return enhanced_embeddings
        
        return embeddings


# 测试代码
if __name__ == "__main__":
    processor = NeuroPianoDataProcessor()
    features = processor.extract_all_features()
    
    # 打印一些提取的特征
    print("\nExtracted NeuroPiano features:")
    for feature_type, feature_data in features.items():
        piece_count = len(feature_data)
        print(f"- {feature_type}: {piece_count} pieces")
    
    # 选择一个乐曲并打印其特征
    if features['scores']:
        sample_piece = next(iter(features['scores'].keys()))
        print(f"\nFeatures for piece '{sample_piece}':")
        piece_features = processor.get_piece_features(sample_piece)
        for feature_type, feature_data in piece_features.items():
            print(f"- {feature_type}: {feature_data}")
        
        # 打印演奏指标
        performance_metrics = processor.get_performance_metrics(sample_piece)
        print(f"\nPerformance metrics for '{sample_piece}':")
        for metric, value in performance_metrics.items():
            print(f"- {metric}: {value}") 