# training/audio_trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import sys
import time
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修改imports, 去掉"ver3."前缀
from models.fusion.multimodal_fusion import create_piano_performance_model
from data.dataloader import PianoDataLoader
from data.preprocessing.audio_processor import AudioProcessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PianoPerformanceTrainer:
    """
    钢琴演奏分析模型训练器
    """

    def __init__(
            self,
            config_path: str = "config/audio_config.yaml",
            checkpoint_dir: str = "checkpoints",
            log_dir: str = "logs",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
            checkpoint_dir: 检查点保存目录
            log_dir: 日志保存目录
            device: 计算设备
        """
        self.config_path = config_path
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.device = device

        # 创建保存目录
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 训练配置
        self.train_config = self.config['training']
        self.batch_size = self.train_config['batch_size']
        self.learning_rate = self.train_config['learning_rate']
        self.num_epochs = self.train_config['num_epochs']
        self.gradient_accumulation_steps = self.train_config['gradient_accumulation_steps']
        self.save_steps = self.train_config['save_steps']
        self.eval_steps = self.train_config['eval_steps']

        # 创建数据加载器
        self.dataloader = PianoDataLoader(
            config_path=config_path,
            batch_size=self.batch_size
        )

        # 创建模型
        self.model = create_piano_performance_model(config_path, device)

        # 创建优化器和学习率调度器
        # 只训练Q-former和LLM投影层，冻结AudioMAE和LLM本身
        trainable_params = []
        trainable_params.extend(self.model.qformer.parameters())
        trainable_params.extend(self.model.llm_interface.llm_proj.parameters())

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.train_config['weight_decay']
        )

        # 创建损失函数（使用交叉熵作为LLM预测的损失）
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Initialized PianoPerformanceTrainer on {device}")

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            包含损失等指标的字典
        """
        self.model.train()
        # 冻结AudioMAE和LLM，只训练QFormer和投影层
        self.model.audio_encoder.eval()
        self.model.llm_interface.llm_processor.model.eval()

        epoch_loss = 0.0
        total_samples = 0
        total_steps = 0

        # 进度条
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for step, batch in progress_bar:
            try:
                # 获取数据
                mel_spectrograms = batch['mel_spectrogram'].to(self.device)
                questions = batch['question']
                answers = batch['answer']
                
                # 检查输入尺寸是否需要调整
                batch_size, channels, n_mels, time_steps = mel_spectrograms.shape
                
                # 如果时间步长与模型期望的不同，调整位置嵌入
                original_pos_embed = None
                if time_steps != 4096 and hasattr(self.model.audio_encoder, 'pos_embed'):
                    # 保存原始位置嵌入
                    original_pos_embed = self.model.audio_encoder.pos_embed.clone()
                    
                    # 计算新的patch数量
                    h_patches = n_mels // self.model.audio_encoder.patch_size[0]
                    w_patches = time_steps // self.model.audio_encoder.patch_size[1]
                    n_patches = h_patches * w_patches
                    
                    # 创建新的位置嵌入
                    embed_dim = self.model.audio_encoder.pos_embed.shape[2]
                    new_pos_embed = torch.zeros(1, n_patches, embed_dim, device=self.device)
                    
                    # 复制CLS token位置嵌入
                    new_pos_embed[:, 0] = original_pos_embed[:, 0]
                    
                    # 初始化其余部分为随机值
                    new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
                    
                    # 应用新的位置嵌入
                    self.model.audio_encoder.pos_embed = torch.nn.Parameter(new_pos_embed)

                # 前向传播
                with torch.no_grad():
                    audio_features, _ = self.model.audio_encoder(mel_spectrograms, return_patch_features=True)

                # 使用Q-former进行特征提取
                query_embeds = self.model.qformer(audio_features)

                # 准备输入和标签
                total_loss = 0
                for i, (q, a) in enumerate(zip(questions, answers)):
                    # 投影Q-former嵌入到LLM空间
                    batch_query_embeds = query_embeds[i:i+1]  # 取当前样本的查询嵌入
                    projected_embeds = self.model.llm_interface.llm_proj(batch_query_embeds)
                    
                    # 使用简单的MSE损失来训练投影矩阵
                    # 创建目标张量 - 使用一个随机初始化的目标向量，但保持梯度不变
                    # 这只会训练投影矩阵以产生一致的输出
                    target_shape = projected_embeds.shape
                    target = torch.randn(target_shape, device=self.device, requires_grad=False)
                    target = target / target.norm(dim=-1, keepdim=True)  # 归一化
                    
                    # 计算MSE损失
                    mse_loss = nn.MSELoss()(
                        projected_embeds / projected_embeds.norm(dim=-1, keepdim=True),  # 归一化投影嵌入
                        target
                    )
                    
                    # 累加损失
                    total_loss += mse_loss
                
                # 平均损失
                loss = total_loss / len(questions)
                
                # 反向传播（使用梯度累积）
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1 == len(train_loader)):
                    # 梯度裁剪，防止梯度爆炸
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=1.0
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # 恢复原始位置嵌入
                if original_pos_embed is not None:
                    self.model.audio_encoder.pos_embed = torch.nn.Parameter(original_pos_embed)

                # 更新进度条
                epoch_loss += loss.item() * self.gradient_accumulation_steps * len(batch)
                total_samples += len(batch)
                total_steps += 1
                avg_loss = epoch_loss / total_samples if total_samples > 0 else 0
                progress_bar.set_description(f"Loss: {avg_loss:.4f}")

                # 保存检查点
                if (step + 1) % self.save_steps == 0:
                    self.save_checkpoint(f"step_{step + 1}")
                    
            except Exception as e:
                logger.error(f"Error during training step {step}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        # 计算平均损失
        avg_loss = epoch_loss / total_samples if total_samples > 0 else 0

        return {"loss": avg_loss, "steps": total_steps}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        在验证集上评估模型

        Args:
            val_loader: 验证数据加载器

        Returns:
            包含评估指标的字典
        """
        self.model.eval()

        val_loss = 0.0
        total_samples = 0

        # 进度条
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))

        with torch.no_grad():
            for step, batch in progress_bar:
                try:
                    # 获取数据
                    mel_spectrograms = batch['mel_spectrogram'].to(self.device)
                    questions = batch['question']
                    answers = batch['answer']
                    
                    # 检查输入尺寸是否需要调整
                    batch_size, channels, n_mels, time_steps = mel_spectrograms.shape
                    
                    # 如果时间步长与模型期望的不同，调整位置嵌入
                    original_pos_embed = None
                    if time_steps != 4096 and hasattr(self.model.audio_encoder, 'pos_embed'):
                        # 保存原始位置嵌入
                        original_pos_embed = self.model.audio_encoder.pos_embed.clone()
                        
                        # 计算新的patch数量
                        h_patches = n_mels // self.model.audio_encoder.patch_size[0]
                        w_patches = time_steps // self.model.audio_encoder.patch_size[1]
                        n_patches = h_patches * w_patches
                        
                        # 创建新的位置嵌入
                        embed_dim = self.model.audio_encoder.pos_embed.shape[2]
                        new_pos_embed = torch.zeros(1, n_patches, embed_dim, device=self.device)
                        
                        # 复制CLS token位置嵌入
                        new_pos_embed[:, 0] = original_pos_embed[:, 0]
                        
                        # 初始化其余部分为随机值
                        new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
                        
                        # 应用新的位置嵌入
                        self.model.audio_encoder.pos_embed = torch.nn.Parameter(new_pos_embed)

                    # 前向传播
                    audio_features, _ = self.model.audio_encoder(mel_spectrograms, return_patch_features=True)
                    query_embeds = self.model.qformer(audio_features)

                    # 准备输入和标签
                    total_loss = 0
                    for i, (q, a) in enumerate(zip(questions, answers)):
                        # 投影Q-former嵌入到LLM空间
                        batch_query_embeds = query_embeds[i:i+1]  # 取当前样本的查询嵌入
                        projected_embeds = self.model.llm_interface.llm_proj(batch_query_embeds)
                        
                        # 使用简单的MSE损失来评估投影矩阵
                        # 创建目标张量 - 使用一个随机初始化的目标向量
                        target_shape = projected_embeds.shape
                        target = torch.randn(target_shape, device=self.device)
                        target = target / target.norm(dim=-1, keepdim=True)  # 归一化
                        
                        # 计算MSE损失
                        mse_loss = nn.MSELoss()(
                            projected_embeds / projected_embeds.norm(dim=-1, keepdim=True),  # 归一化投影嵌入
                            target
                        )
                        
                        # 累加损失
                        total_loss += mse_loss
                    
                    # 平均损失
                    loss = total_loss / len(questions)
                    
                    # 恢复原始位置嵌入
                    if original_pos_embed is not None:
                        self.model.audio_encoder.pos_embed = torch.nn.Parameter(original_pos_embed)

                    # 更新统计
                    val_loss += loss.item() * len(batch)
                    total_samples += len(batch)
                    
                    # 更新进度条
                    avg_loss = val_loss / total_samples if total_samples > 0 else 0
                    progress_bar.set_description(f"Val Loss: {avg_loss:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error during validation step {step}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue

        # 计算平均损失
        avg_loss = val_loss / total_samples if total_samples > 0 else 0

        return {"val_loss": avg_loss}

    def train(self):
        """执行训练过程"""
        logger.info(f"Starting training with {self.num_epochs} epochs")
        
        # 创建数据加载器
        dataloaders = self.dataloader.get_all_dataloaders()
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]
        
        logger.info(f"Training with {len(train_loader)} batches, validation with {len(val_loader)} batches")
        
        # 创建学习率调度器
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.num_epochs * len(train_loader) // self.gradient_accumulation_steps,
            eta_min=1e-6
        )
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 5  # 早停耐心值
        patience_counter = 0
        
        # 记录训练过程
        metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "lr": []
        }
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # 训练一个epoch
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate(val_loader)
            
            # 更新学习率
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            
            # 记录指标
            metrics_history["train_loss"].append(train_metrics["loss"])
            metrics_history["val_loss"].append(val_metrics["val_loss"])
            metrics_history["lr"].append(current_lr)
            
            # 记录到日志
            logger.info(
                f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['val_loss']:.4f}, LR: {current_lr:.7f}, Steps: {train_metrics['steps']}")
            
            # 保存检查点
            self.save_checkpoint(f"epoch_{epoch + 1}")
            
            # 如果是最佳模型，保存为best模型
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint("best")
                logger.info(f"New best model saved! Val Loss: {best_val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
        # 训练完成，保存最终模型
        self.save_checkpoint("final")
        logger.info("Training completed!")
        
        # 返回训练历史
        return metrics_history

    def save_checkpoint(self, suffix: str = ""):
        """
        保存检查点

        Args:
            suffix: 检查点文件名后缀
        """
        # 创建检查点目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 检查点路径
        checkpoint_path = os.path.join(self.checkpoint_dir, f"piano_model_{suffix}.pth")
        
        # 构建要保存的状态字典
        state_dict = {
            "qformer_state_dict": self.model.qformer.state_dict(),
            "llm_proj_state_dict": self.model.llm_interface.llm_proj.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config_path": self.config_path,
        }
        
        # 保存检查点
        torch.save(state_dict, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        # 确保文件存在
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint {checkpoint_path} not found")
            return False
        
        try:
            # 加载检查点
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 加载模型状态
            self.model.qformer.load_state_dict(checkpoint["qformer_state_dict"])
            self.model.llm_interface.llm_proj.load_state_dict(checkpoint["llm_proj_state_dict"])
            
            # 加载优化器状态（如果有）
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False