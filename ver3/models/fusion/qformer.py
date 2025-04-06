import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import sys
from transformers import BertConfig, BertModel, BertLMHeadModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QFormerConfig(BertConfig):
    """Q-former配置类，扩展自BertConfig"""
    model_type = "qformer"

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            cross_attention_frequency=2,
            encoder_hidden_size=1024,
            **kwargs
    ):
        """
        初始化QFormerConfig

        Args:
            vocab_size: 词汇表大小
            hidden_size: 隐藏层维度
            num_hidden_layers: 隐藏层数量
            num_attention_heads: 注意力头数
            intermediate_size: 中间层维度
            hidden_act: 隐藏层激活函数
            hidden_dropout_prob: 隐藏层dropout比率
            attention_probs_dropout_prob: 注意力概率dropout比率
            max_position_embeddings: 最大位置嵌入数
            type_vocab_size: 类型词汇表大小
            initializer_range: 初始化范围
            layer_norm_eps: LayerNorm epsilon
            pad_token_id: 填充token ID
            position_embedding_type: 位置嵌入类型
            cross_attention_frequency: 交叉注意力频率
            encoder_hidden_size: 编码器隐藏层维度
        """
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            position_embedding_type=position_embedding_type,
            **kwargs,
        )
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size


class QFormerMultiHeadAttention(nn.Module):
    """多头注意力模块，支持交叉注意力"""

    def __init__(
            self,
            hidden_size: int,
            num_attention_heads: int,
            dropout: float = 0.0,
            is_cross_attention: bool = False
    ):
        """
        初始化QFormer多头注意力

        Args:
            hidden_size: 隐藏层维度
            num_attention_heads: 注意力头数
            dropout: Dropout比率
            is_cross_attention: 是否为交叉注意力
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.is_cross_attention = is_cross_attention

        # 查询、键、值投影
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # 输出投影和dropout
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        重塑张量以便进行多头注意力计算

        Args:
            x: 输入张量 [B, N, D]

        Returns:
            重塑后的张量 [B, num_heads, N, head_size]
        """
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            hidden_states: 输入隐藏状态 [B, N, D]
            attention_mask: 注意力掩码 [B, 1, N, N]
            encoder_hidden_states: 编码器隐藏状态 (用于交叉注意力)
            encoder_attention_mask: 编码器注意力掩码
            output_attentions: 是否输出注意力权重

        Returns:
            输出隐藏状态, 可选的注意力权重
        """
        # 使用自注意力或交叉注意力，计算查询
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # 对于交叉注意力，键和值来自编码器隐藏状态
        if self.is_cross_attention and encoder_hidden_states is not None:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        # 对于自注意力，键和值来自输入隐藏状态
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 归一化注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)

        # 输出投影
        output = self.output(context_layer)

        if output_attentions:
            return output, attention_probs
        else:
            return (output,)


class QFormerLayer(nn.Module):
    """Q-former层，支持自注意力和交叉注意力"""

    def __init__(
            self,
            config: QFormerConfig,
            layer_idx: int
    ):
        """
        初始化QFormer层

        Args:
            config: QFormer配置
            layer_idx: 层索引
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention = QFormerMultiHeadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob
        )
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

        # 如果需要交叉注意力
        is_cross_attention_layer = layer_idx % config.cross_attention_frequency == 0
        if is_cross_attention_layer:
            self.crossattention = QFormerMultiHeadAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                is_cross_attention=True
            )
            self.layernorm_cross = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            encoder_hidden_states: 编码器隐藏状态
            encoder_attention_mask: 编码器注意力掩码
            output_attentions: 是否输出注意力权重

        Returns:
            输出隐藏状态, 可选的注意力权重
        """
        # 自注意力
        residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = self_attention_outputs[0]
        hidden_states = residual + self.dropout(hidden_states)

        # 交叉注意力
        is_cross_attention_layer = self.layer_idx % self.config.cross_attention_frequency == 0
        if is_cross_attention_layer and encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.layernorm_cross(hidden_states)
            cross_attention_outputs = self.crossattention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = cross_attention_outputs[0]
            hidden_states = residual + self.dropout(hidden_states)

        # 前馈网络
        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attention_outputs[1],)
            if is_cross_attention_layer and encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)

        return outputs


class QFormerEncoder(nn.Module):
    """Q-former编码器"""

    def __init__(self, config: QFormerConfig):
        """
        初始化QFormer编码器

        Args:
            config: QFormer配置
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            QFormerLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            hidden_states: 输入隐藏状态
            attention_mask: 注意力掩码
            encoder_hidden_states: 编码器隐藏状态
            encoder_attention_mask: 编码器注意力掩码
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态
            return_dict: 是否返回字典

        Returns:
            输出隐藏状态, 可选的所有隐藏状态, 可选的注意力权重
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)
                # 如果有交叉注意力结果，添加它们
                if len(layer_outputs) > 2:
                    all_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

        return hidden_states, all_hidden_states, all_attentions


class QFormerModel(nn.Module):
    """Q-former模型"""

    def __init__(self, config: QFormerConfig):
        """
        初始化QFormer模型

        Args:
            config: QFormer配置
        """
        super().__init__()
        self.config = config

        # 查询嵌入
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.hidden_size)
        )
        self.query_positions = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.hidden_size)
        )

        # 编码器投影
        if hasattr(config, "encoder_hidden_size"):
            self.encoder_proj = nn.Linear(
                config.encoder_hidden_size, config.hidden_size
            )
        else:
            self.encoder_proj = None

        # 模型组件
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = QFormerEncoder(config)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 初始化查询令牌和位置
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        nn.init.trunc_normal_(self.query_positions, std=0.02)

        # 初始化编码器投影（如果存在）
        if self.encoder_proj is not None:
            nn.init.trunc_normal_(self.encoder_proj.weight, std=0.02)
            if self.encoder_proj.bias is not None:
                nn.init.constant_(self.encoder_proj.bias, 0)

    def forward(
            self,
            query_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            query_embeds: 查询嵌入，如果为None则使用默认查询令牌
            encoder_hidden_states: 编码器隐藏状态
            encoder_attention_mask: 编码器注意力掩码
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态
            return_dict: 是否返回字典

        Returns:
            查询特征, 可选的所有隐藏状态, 可选的注意力权重
        """
        batch_size = encoder_hidden_states.size(0) if encoder_hidden_states is not None else 1

        # 初始化查询特征
        if query_embeds is None:
            query_embeds = self.query_tokens.expand(batch_size, -1, -1)

        # 添加位置信息
        hidden_states = query_embeds + self.query_positions

        # 应用LayerNorm和dropout
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # 投影编码器隐藏状态（如果需要）
        if encoder_hidden_states is not None and self.encoder_proj is not None:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)

        # 创建查询注意力掩码（通常是None，因为我们不掩盖查询令牌之间的注意力）
        query_attention_mask = None

        # 传递到编码器
        outputs = self.encoder(
            hidden_states=hidden_states,
            attention_mask=query_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return outputs


class AudioQFormer(nn.Module):
    """
    AudioQFormer模型，用于连接音频特征和LLM
    """

    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            num_query_tokens: int = 32,
            cross_attention_freq: int = 2
    ):
        """
        初始化AudioQFormer

        Args:
            config_path: 配置文件路径
            num_query_tokens: 查询令牌数量
            cross_attention_freq: 交叉注意力频率
        """
        super().__init__()

        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            qformer_config = config['qformer']
            audio_config = config['audio_model']
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. This should not happen as the file should have been created already.")
            # 创建一个默认配置
            qformer_config = {
                'hidden_size': 768,
                'num_hidden_layers': 4,
                'num_attention_heads': 12,
            }
            audio_config = {
                'feature_dim': 1024,
            }

        # 创建QFormer配置
        self.config = QFormerConfig(
            encoder_hidden_size=audio_config['feature_dim'],
            hidden_size=qformer_config['hidden_size'],
            num_hidden_layers=qformer_config['num_hidden_layers'],
            num_attention_heads=qformer_config['num_attention_heads'],
            intermediate_size=qformer_config['hidden_size'] * 4,
            cross_attention_frequency=cross_attention_freq
        )

        # 设置查询令牌数量
        self.config.num_query_tokens = num_query_tokens

        # 创建QFormer模型
        self.qformer = QFormerModel(self.config)

        logger.info(f"Initialized AudioQFormer with {num_query_tokens} query tokens")

    def forward(
            self,
            audio_features: torch.Tensor,
            output_attentions: bool = False,
            output_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            audio_features: 音频特征 [B, N, D]
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出所有隐藏状态

        Returns:
            查询特征, 可选的所有隐藏状态, 可选的注意力权重
        """
        # 调用QFormer
        outputs = self.qformer(
            encoder_hidden_states=audio_features,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )

        # 获取查询特征
        query_output = outputs[0]  # [B, num_query_tokens, hidden_size]

        return query_output


def create_audio_qformer(config_path: str = "config/audio_config.yaml") -> AudioQFormer:
    """
    创建音频Q-former模型

    Args:
        config_path: 配置文件路径

    Returns:
        AudioQFormer实例
    """
    try:
        # 检查配置文件是否存在
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, creating default config")
            # 如果配置文件不存在，创建默认配置
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
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
                    'hidden_size': 768,
                    'num_attention_heads': 12,
                    'num_hidden_layers': 4,
                    'num_query_tokens': 32
                }
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
    except Exception as e:
        logger.error(f"Error handling config file: {e}")
        raise

    # 创建Q-former模型
    model = AudioQFormer(
        config_path=config_path,
        num_query_tokens=32,
        cross_attention_freq=2
    )

    return model


# 测试代码
if __name__ == "__main__":
    # 创建配置文件目录（如果不存在）
    os.makedirs("config", exist_ok=True)

    # 如果配置文件不存在，创建一个临时配置
    if not os.path.exists("config/audio_config.yaml"):
        with open("config/audio_config.yaml", "w") as f:
            f.write("""
audio_model:
  feature_dim: 1024

qformer:
  hidden_size: 768
  num_attention_heads: 12
  num_hidden_layers: 4
  num_query_tokens: 32
            """)

    # 创建模型
    model = create_audio_qformer()
    print(f"Created AudioQFormer model")

    # 测试前向传播
    batch_size = 2
    seq_len = 64  # 音频特征序列长度
    feature_dim = 1024  # 音频特征维度

    # 创建随机音频特征
    audio_features = torch.randn(batch_size, seq_len, feature_dim)

    # 前向传播
    with torch.no_grad():
        query_output = model(audio_features)

        print(f"Query output shape: {query_output.shape}")
        # 应该是 [batch_size, num_query_tokens, hidden_size]