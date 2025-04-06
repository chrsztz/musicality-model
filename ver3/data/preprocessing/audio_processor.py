import os
import torch
import torchaudio
import numpy as np
import librosa
from typing import Dict, Tuple, Optional, Union
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProcessor:
    """音频处理类，用于钢琴演奏音频的预处理"""
    
    def __init__(self, config_path: str = "config/audio_config.yaml"):
        """
        初始化音频处理器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Successfully loaded audio configuration from {config_path}")
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, creating default audio config")
            # 创建配置文件目录（如果不存在）
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            # 创建默认配置
            config = {
                'audio_model': {
                    'sample_rate': 32000,
                    'n_mels': 128,
                    'max_length': 4096,
                    'frameshift': 10
                }
            }
            
            # 写入配置文件
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f)
            
        self.audio_config = config['audio_model']
        self.sample_rate = self.audio_config['sample_rate']
        self.n_mels = self.audio_config['n_mels']
        self.max_length = self.audio_config['max_length']
        self.frameshift = self.audio_config['frameshift'] / 1000  # 转换为秒
        
        logger.info(f"Initialized AudioProcessor with sample_rate={self.sample_rate}, n_mels={self.n_mels}")
        
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频张量和采样率
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            waveform, sr = torchaudio.load(audio_path)
            # 转为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            # 重采样到目标采样率
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            return waveform, self.sample_rate
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise
    
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        提取Mel频谱图特征
        
        Args:
            waveform: 音频波形张量 [1, T]
            
        Returns:
            Mel频谱图特征 [1, n_mels, T']
        """
        # 使用更大的FFT窗口以支持更多的梅尔频带
        n_fft = 2048  # 固定为2048点，这是一个常用值，可以支持128个梅尔频带
        hop_length = int(self.frameshift * self.sample_rate)
        
        # 计算Nyquist频率（采样率的一半）
        nyquist = self.sample_rate // 2
        
        # 使用torchaudio提取梅尔频谱，设置明确的频率范围
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=self.n_mels,
            f_min=0.0,
            f_max=nyquist,  # 设置为Nyquist频率
            power=2.0,
            norm="slaney",  # 使用Slaney归一化方法
            mel_scale="htk"  # 使用HTK梅尔刻度
        )(waveform)
        
        # 转换为对数刻度
        log_mel_spectrogram = torch.log(mel_spectrogram + 1e-9)
        
        return log_mel_spectrogram
    
    def normalize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        标准化频谱图
        
        Args:
            spectrogram: Mel频谱图特征
            
        Returns:
            标准化后的Mel频谱图
        """
        mean = spectrogram.mean()
        std = spectrogram.std()
        normalized_spectrogram = (spectrogram - mean) / (std + 1e-9)
        return normalized_spectrogram
    
    def pad_or_truncate(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        填充或截断频谱图到固定长度
        
        Args:
            spectrogram: Mel频谱图特征 [1, n_mels, T]
            
        Returns:
            处理后的频谱图 [1, n_mels, max_length]
        """
        _, _, time_steps = spectrogram.shape
        
        if time_steps > self.max_length:
            # 截断
            spectrogram = spectrogram[:, :, :self.max_length]
        elif time_steps < self.max_length:
            # 填充
            padding = torch.zeros(1, self.n_mels, self.max_length - time_steps)
            spectrogram = torch.cat([spectrogram, padding], dim=2)
            
        return spectrogram
    
    def process_audio(self, audio_path: str) -> Dict[str, torch.Tensor]:
        """
        处理音频文件，提取特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            包含处理后特征的字典
        """
        waveform, _ = self.load_audio(audio_path)
        mel_spectrogram = self.extract_mel_spectrogram(waveform)
        normalized_spectrogram = self.normalize_spectrogram(mel_spectrogram)
        padded_spectrogram = self.pad_or_truncate(normalized_spectrogram)
        
        return {
            "waveform": waveform,
            "mel_spectrogram": padded_spectrogram,
            "audio_path": audio_path
        }
    
    def process_audio_batch(self, audio_paths: list) -> Dict[str, torch.Tensor]:
        """
        批量处理音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            包含批量处理后特征的字典
        """
        batch_spectrograms = []
        batch_waveforms = []
        
        for audio_path in audio_paths:
            features = self.process_audio(audio_path)
            batch_spectrograms.append(features["mel_spectrogram"])
            batch_waveforms.append(features["waveform"])
            
        return {
            "waveforms": torch.cat(batch_waveforms, dim=0),
            "mel_spectrograms": torch.cat(batch_spectrograms, dim=0),
            "audio_paths": audio_paths
        }


# 测试代码
if __name__ == "__main__":
    # 创建配置文件目录（如果不存在）
    os.makedirs("config", exist_ok=True)
    
    # 如果配置文件不存在，创建一个临时配置
    if not os.path.exists("config/audio_config.yaml"):
        with open("config/audio_config.yaml", "w") as f:
            f.write("""
audio_model:
  sample_rate: 32000
  n_mels: 128
  max_length: 4096
  frameshift: 10
            """)
    
    processor = AudioProcessor()
    
    # 测试音频处理
    test_audio_path = "path/to/test_audio.wav"  # 替换为实际测试音频路径
    
    if os.path.exists(test_audio_path):
        features = processor.process_audio(test_audio_path)
        print(f"Processed audio: {features['audio_path']}")
        print(f"Waveform shape: {features['waveform'].shape}")
        print(f"Mel spectrogram shape: {features['mel_spectrogram'].shape}")
    else:
        print(f"Test audio file not found: {test_audio_path}")
        print("Please provide a valid audio path for testing.")