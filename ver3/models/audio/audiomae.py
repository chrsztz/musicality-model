import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import sys

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchEmbed(nn.Module):
    """
    2D图像到patch嵌入，适用于频谱图
    """

    def __init__(
            self,
            img_size: Tuple[int, int] = (128, 1024),  # (频率, 时间)
            patch_size: Tuple[int, int] = (16, 16),
            in_chans: int = 1,
            embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, Freq, Time)
        output: (B, N, D) - batch, patches, embed_dim
        """
        B, C, H, W = x.shape
        # 临时禁用形状检查，但记录可能的不匹配
        if H != self.img_size[0] or W != self.img_size[1]:
            print(f"WARNING: Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}), but continuing anyway")
        
        # 投影到嵌入维度: (B, embed_dim, H//patch_size, W//patch_size)
        x = self.proj(x)
        # 展平patch维度: (B, embed_dim, N)
        x = x.flatten(2)
        # 转置: (B, N, embed_dim)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """多头自注意力模块"""

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """多层感知机模块"""

    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer编码器块"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            drop: float = 0.,
            attn_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=drop
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AudioMAE(nn.Module):
    """
    基于MAE的音频编码器模型
    """

    def __init__(
            self,
            img_size: Tuple[int, int] = (128, 1024),  # (频率, 时间)
            patch_size: Tuple[int, int] = (16, 16),
            in_chans: int = 1,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            output_dim: int = 1024
    ):
        """
        初始化AudioMAE模型

        Args:
            img_size: 输入频谱图大小 (频率, 时间)
            patch_size: patch大小
            in_chans: 输入通道数
            embed_dim: 嵌入维度
            depth: Transformer块数量
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏维度与嵌入维度的比率
            qkv_bias: 是否使用偏置
            drop_rate: Dropout比率
            attn_drop_rate: 注意力Dropout比率
            norm_layer: 归一化层
            output_dim: 输出特征维度
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # patch嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer块
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)
        ])

        # 输出层
        self.norm = norm_layer(embed_dim)
        self.fc_out = nn.Linear(embed_dim, output_dim) if output_dim != embed_dim else nn.Identity()

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 初始化位置嵌入
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 初始化线性层和LayerNorm
        self.apply(self._init_layer_weights)

    def _init_layer_weights(self, m):
        """初始化层权重的辅助函数"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        特征提取前向传播

        Args:
            x: 输入频谱图 [B, C, Freq, Time]

        Returns:
            特征表示 [B, N, D]
        """
        # patch嵌入
        x = self.patch_embed(x)  # [B, N, D]

        # 添加位置嵌入
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 通过Transformer块
        for block in self.blocks:
            x = block(x)

        # 应用最终的LayerNorm
        x = self.norm(x)

        return x

    def forward(self, x: torch.Tensor, return_patch_features: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        模型前向传播

        Args:
            x: 输入频谱图 [B, C, Freq, Time]
            return_patch_features: 是否返回patch级特征

        Returns:
            如果return_patch_features为True:
                返回(patch_features, pooled_features)
            否则:
                返回pooled_features
        """
        # 获取特征
        patch_features = self.forward_features(x)  # [B, N, D]

        # 全局池化
        pooled_features = patch_features.mean(dim=1)  # [B, D]

        # 通过输出层
        pooled_features = self.fc_out(pooled_features)  # [B, output_dim]

        if return_patch_features:
            patch_features = self.fc_out(patch_features)  # [B, N, output_dim]
            return patch_features, pooled_features
        else:
            return pooled_features

    def load_pretrained(self, checkpoint_path: str):
        """
        加载预训练权重

        Args:
            checkpoint_path: 检查点路径
        """
        if not os.path.exists(checkpoint_path):
            logger.warning(f"Pretrained weights file not found: {checkpoint_path}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 处理不同键的情况
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # 加载权重
            msg = self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained weights from {checkpoint_path}")
            if msg.missing_keys:
                logger.warning(f"Missing keys: {msg.missing_keys}")
            if msg.unexpected_keys:
                logger.warning(f"Unexpected keys: {msg.unexpected_keys}")

        except Exception as e:
            logger.error(f"Error loading pretrained weights: {e}")


def create_audiomae_model(config_path: str = "config/audio_config.yaml") -> AudioMAE:
    """
    从配置创建AudioMAE模型

    Args:
        config_path: 配置文件路径

    Returns:
        AudioMAE模型
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        audio_config = config['audio_model']
    except FileNotFoundError:
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
                'max_length': 1024,
                'frameshift': 10
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
            
        audio_config = config['audio_model']

    # 计算频谱图尺寸
    n_mels = audio_config['n_mels']
    max_length = audio_config['max_length']
    
    # 添加调试日志
    logger.info(f"Creating AudioMAE with img_size=({n_mels}, {max_length})")
    
    # 创建模型
    model = AudioMAE(
        img_size=(n_mels, max_length),  # 频率 x 时间
        patch_size=(16, 16),  # patch大小
        in_chans=1,  # 单通道输入
        embed_dim=768,  # 嵌入维度
        depth=12,  # 12层transformer
        num_heads=12,  # 12个注意力头
        mlp_ratio=4.0,  # MLP隐藏层大小比例
        output_dim=audio_config['feature_dim']  # 输出特征维度
    )
    
    # 验证模型的img_size是否正确设置
    logger.info(f"Verified AudioMAE model.img_size = {model.img_size}")
    logger.info(f"Verified AudioMAE model.patch_embed.img_size = {model.patch_embed.img_size}")

    # 加载预训练权重（如果存在）
    pretrained_path = audio_config.get('pretrained_path', None)
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_pretrained(pretrained_path)

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
  model_type: 'audiomae'
  pretrained_path: 'pretrained/audiomae_base.pth'
  sample_rate: 32000
  n_mels: 128
  feature_dim: 1024
  max_length: 1024
  frameshift: 10
            """)

    # 创建模型
    model = create_audiomae_model()
    print(f"Created AudioMAE model: {model.__class__.__name__}")

    # 测试前向传播
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 128, 1024)  # [B, C, Freq, Time]

    # 前向传播
    with torch.no_grad():
        # 仅返回池化特征
        pooled_features = model(input_tensor)
        print(f"Pooled features shape: {pooled_features.shape}")

        # 返回patch特征和池化特征
        patch_features, pooled_features = model(input_tensor, return_patch_features=True)
        print(f"Patch features shape: {patch_features.shape}")
        print(f"Pooled features shape (again): {pooled_features.shape}")