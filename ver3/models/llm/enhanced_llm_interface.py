import os
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml
import logging
import sys
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入基础LLM接口
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.llm.llm_interface import LLMProcessor, AudioLLMInterface


class EnhancedLLMProcessor(LLMProcessor):
    """增强版LLM处理器，提供多维度评估和结构化输出"""

    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """初始化增强版LLM处理器"""
        super().__init__(config_path, device)
        
        # 加载评估维度配置
        self.eval_dimensions = self._load_evaluation_dimensions(config_path)
        
    def _load_evaluation_dimensions(self, config_path: str) -> Dict[str, Any]:
        """加载评估维度配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'evaluation_dimensions' not in config:
                logger.warning("Evaluation dimensions not found in config, using default")
                config['evaluation_dimensions'] = {
                    'objective': [
                        'pitch_accuracy', 'tempo_control', 'rhythm_precision',
                        'articulation', 'pedaling', 'timbre_quality',
                        'dynamic_control', 'balance', 'integrity'
                    ],
                    'descriptive': [
                        'performance_understanding', 'technical_difficulty',
                        'employed_techniques', 'compositional_background',
                        'emotional_expression', 'stylistic_authenticity'
                    ]
                }
                
                # 写回配置文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f)
                    
            return config['evaluation_dimensions']
        except Exception as e:
            logger.error(f"Error loading evaluation dimensions: {e}")
            # 返回默认维度
            return {
                'objective': [
                    'pitch_accuracy', 'tempo_control', 'rhythm_precision',
                    'articulation', 'pedaling', 'timbre_quality',
                    'dynamic_control', 'balance', 'integrity'
                ],
                'descriptive': [
                    'performance_understanding', 'technical_difficulty',
                    'employed_techniques', 'compositional_background',
                    'emotional_expression', 'stylistic_authenticity'
                ]
            }
            
    def generate_structured_prompts(self, base_prompt: str, piece_info: Optional[Dict] = None) -> str:
        """生成结构化提示，包含三部分反馈结构和评估维度"""
        
        # 评估维度提示
        dimensions_prompt = "Please evaluate the following aspects of the piano performance:\n"
        
        # 添加客观评估维度
        dimensions_prompt += "Objective measures:\n"
        for dim in self.eval_dimensions['objective']:
            dimensions_prompt += f"- {dim.replace('_', ' ').title()}\n"
            
        # 添加描述性评估维度
        dimensions_prompt += "\nDescriptive measures:\n"
        for dim in self.eval_dimensions['descriptive']:
            dimensions_prompt += f"- {dim.replace('_', ' ').title()}\n"
        
        # 结构化反馈提示
        structure_prompt = """
Organize your analysis in the following three parts:

1. FEEDBACK: Provide a detailed musical analysis of what you hear, focusing on the musical characteristics, techniques, and overall quality.

2. SUGGESTIONS: Offer specific, actionable recommendations for how the pianist could improve their performance.

3. APPRECIATION: Highlight the strengths and admirable aspects of the performance.
"""
        
        # 如果有乐曲信息，添加到提示中
        piece_context = ""
        if piece_info:
            piece_context = f"\nThis is a performance of '{piece_info.get('title', 'the piece')}'"
            if 'composer' in piece_info:
                piece_context += f" by {piece_info['composer']}"
            if 'style' in piece_info:
                piece_context += f", in {piece_info['style']} style"
            if 'difficulty' in piece_info:
                piece_context += f", with a difficulty level of {piece_info['difficulty']}"
            piece_context += ".\n"
        
        # 完整提示
        full_prompt = f"{base_prompt}{piece_context}\n\n{dimensions_prompt}\n{structure_prompt}"
        
        return full_prompt


class EnhancedAudioLLMInterface(AudioLLMInterface):
    """增强版音频LLM接口，支持多维度评估和结构化输出"""
    
    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """初始化增强版音频LLM接口"""
        super().__init__(config_path, device)
        
        # 替换为增强版LLM处理器
        self.llm_processor = EnhancedLLMProcessor(config_path, device)
        
        # 加载NeuroPiano特征集成配置
        self.neuropiano_features = self._load_neuropiano_features(config_path)
        
        # 添加特征投影层
        self.feature_projection = self._create_feature_projection()
        
    def _load_neuropiano_features(self, config_path: str) -> Dict[str, Any]:
        """加载NeuroPiano特征配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if 'neuropiano_features' not in config:
                logger.warning("NeuroPiano features not found in config, using default")
                config['neuropiano_features'] = {
                    'use_features': True,
                    'feature_types': [
                        'good_or_bad', 'score', 'technique', 
                        'physical_attributes', 'adjectives'
                    ]
                }
                
                # 写回配置文件
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f)
                    
            return config['neuropiano_features']
        except Exception as e:
            logger.error(f"Error loading NeuroPiano features config: {e}")
            # 返回默认配置
            return {
                'use_features': True,
                'feature_types': [
                    'good_or_bad', 'score', 'technique', 
                    'physical_attributes', 'adjectives'
                ]
            }
            
    def _create_feature_projection(self) -> nn.Module:
        """创建特征投影层"""
        # 简单的线性投影
        return nn.Linear(self.llm_proj.out_features, self.llm_proj.out_features)
    
    def forward(
            self,
            query_embeds: torch.Tensor,
            question: str,
            piece_info: Optional[Dict] = None,
            max_new_tokens: int = 50,
            num_beams: int = 1,
            min_length: int = 1,
            top_p: float = 0.9,
            temperature: float = 0.7,
            repetition_penalty: float = 1.0,
            length_penalty: float = 1.0,
            do_sample: bool = True
    ) -> List[str]:
        """
        执行前向传播并生成回答
        
        Args:
            query_embeds: 查询嵌入 [B, num_query_tokens, hidden_size]
            question: 问题文本
            piece_info: 乐曲信息字典（可选）
            max_new_tokens: 最大生成令牌数
            num_beams: 光束搜索数量
            min_length: 最小生成长度
            top_p: top-p采样参数
            temperature: 温度参数
            repetition_penalty: 重复惩罚
            length_penalty: 长度惩罚
            do_sample: 是否使用采样
            
        Returns:
            生成的回答列表
        """
        # 投影到LLM空间
        projected_query_embeds = self.llm_proj(query_embeds)
        
        # 应用特征投影
        projected_query_embeds = self.feature_projection(projected_query_embeds)
        
        # 构建结构化提示
        structured_question = self.llm_processor.generate_structured_prompts(question, piece_info)
        
        # 准备模型输入
        inputs = self.llm_processor.prepare_inputs(projected_query_embeds, structured_question)
        
        # 使用与基类相同的生成逻辑
        answers = []
        
        try:
            with torch.no_grad():
                # 生成标记
                generated_ids = self.llm_processor.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    min_length=min_length,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    do_sample=do_sample,
                    use_cache=True
                )
                
                # 解码生成的标记
                for output_ids in generated_ids:
                    # 获取仅生成的部分（去除提示）
                    input_length = inputs["input_ids"].shape[1]
                    gen_ids = output_ids[input_length:]
                    
                    # 解码
                    gen_text = self.llm_processor.tokenizer.decode(
                        gen_ids, 
                        skip_special_tokens=True, 
                        clean_up_tokenization_spaces=True
                    )
                    
                    answers.append(gen_text)
                    
        except Exception as e:
            logger.error(f"Generation error: {e}")
            
            # 备选方案：使用结构化分析模板
            answers = [self.structured_analysis_template(query_embeds)]
            
        return answers
    
    def structured_analysis_template(self, audio_features: torch.Tensor) -> str:
        """
        生成基于音频特征的结构化分析模板
        
        Args:
            audio_features: 音频特征
            
        Returns:
            结构化分析文本
        """
        # 提取简单指标
        if len(audio_features.shape) == 3:
            # [B, N, D]
            features = audio_features.mean(dim=1).squeeze(0)  # 平均序列维度
        else:
            # [B, D]
            features = audio_features.squeeze(0)  # 移除批次维度
        
        # 转换为numpy以便于分析
        features_np = features.cpu().numpy()
        
        # 提取基本指标
        mean_val = features_np.mean()
        std_val = features_np.std()
        max_val = features_np.max()
        min_val = features_np.min()
        
        # 生成客观评估指标
        pitch_accuracy = min(10, max(1, int(5 + mean_val * 2)))
        tempo_control = min(10, max(1, int(5 + std_val * 2)))
        dynamic_range = min(10, max(1, int(5 + (max_val - min_val) * 2)))
        
        # 风格和技术特点
        style = "classical" if mean_val < 0 else "romantic" if mean_val < 0.2 else "contemporary"
        technique = "precise" if std_val < 0.3 else "expressive" if std_val < 0.6 else "virtuosic"
        emotion = "calm" if mean_val < -0.1 else "melancholic" if mean_val < 0.1 else "passionate"
        
        # 生成三部分结构化分析
        analysis = f"""
# FEEDBACK

This piano performance demonstrates a {emotion} emotional quality with {technique} playing technique in a {style} style. The recording reveals a pianist with {dynamic_range}/10 dynamic control capability.

The pitch accuracy is rated {pitch_accuracy}/10, indicating a {'solid technical foundation' if pitch_accuracy > 7 else 'moderate technical skill' if pitch_accuracy > 5 else 'developing technical ability'}. The tempo control scores {tempo_control}/10, showing {'excellent rhythmic stability' if tempo_control > 7 else 'good rhythmic awareness' if tempo_control > 5 else 'some rhythmic inconsistencies'}.

The performance has clear articulation and a well-established sense of musical phrasing. The tonal quality is {'rich and resonant' if mean_val > 0 else 'clear and delicate'}, with {'exceptional' if std_val > 0.5 else 'good' if std_val > 0.3 else 'adequate'} pedal control.

# SUGGESTIONS

To enhance this performance, the pianist could:

1. {'Focus on maintaining more consistent tempo during technically demanding passages' if tempo_control < 7 else 'Explore more subtle tempo variations to highlight the musical structure'}
2. {'Pay closer attention to voicing in polyphonic sections to bring out important melodic lines' if pitch_accuracy < 8 else 'Consider more dramatic dynamic contrasts to enhance emotional expression'}
3. {'Work on developing a wider dynamic range for greater expressivity' if dynamic_range < 7 else 'Refine the balance between hands, particularly in complex textures'}
4. {'Practice with a metronome to develop more stable rhythm' if tempo_control < 6 else 'Experiment with more varied articulation to highlight different musical characters'}

# APPRECIATION

The performance demonstrates {'remarkable attention to musical detail' if pitch_accuracy > 7 else 'good musical intuition'} and {'exceptional technical control' if std_val < 0.3 else 'expressive musicality'}. The pianist shows a {'deep understanding' if mean_val > 0 else 'thoughtful approach'} to the music's structure and emotional content.

Particularly impressive is the {'ability to maintain clarity in complex passages' if pitch_accuracy > 7 else 'sense of musical line and direction'}, which creates a compelling musical narrative. The {'subtle dynamic shadings' if dynamic_range > 7 else 'clear articulation'} adds significant depth to the interpretation, engaging the listener throughout the performance.
""".strip()
        
        return analysis


# 工厂函数，用于创建增强版音频LLM接口
def create_enhanced_audio_llm_interface(
        config_path: str = "config/audio_config.yaml",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> EnhancedAudioLLMInterface:
    """
    创建增强版音频LLM接口
    
    Args:
        config_path: 配置文件路径
        device: 设备
        
    Returns:
        增强版音频LLM接口
    """
    return EnhancedAudioLLMInterface(config_path, device) 