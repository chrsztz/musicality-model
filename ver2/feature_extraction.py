import numpy as np
import librosa
import pretty_midi
from scipy.stats import skew, kurtosis
import pandas as pd
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings('ignore')


class PianoFeatureExtractor:
    """
    钢琴演奏特征提取器
    提取不同层次的特征，包括低级(低层次)特征和高级(高层次)特征
    """

    def __init__(self):
        """初始化特征提取器"""
        # 特征组标识
        self.feature_groups = {
            'timing': 'low',  # 节奏时值特征 - 低层次
            'articulation': 'low',  # 演奏连贯性特征 - 低层次
            'pedal': 'mid-low',  # 踏板使用特征 - 中低层次
            'timbre': 'mid-low',  # 音色特征 - 中低层次
            'dynamics': 'mid-high',  # 力度特征 - 中高层次
            'music_making': 'mid-high',  # 音乐构建特征 - 中高层次
            'emotion': 'high',  # 情感表达特征 - 高层次
            'interpretation': 'high'  # 演绎特征 - 高层次
        }

    def extract_timing_features(self, midi_data):
        """
        提取节奏时值特征（低层次）

        Args:
            midi_data: pretty_midi.PrettyMIDI对象

        Returns:
            timing_features: 节奏时值特征字典
        """
        # 初始化特征字典
        timing_features = {}

        # 如果没有有效数据，返回空特征
        if midi_data is None or len(midi_data.instruments) == 0:
            return {
                'beat_stability': 0,
                'tempo_mean': 0,
                'tempo_std': 0,
                'rhythm_regularity': 0,
                'timing_accuracy': 0
            }

        # 获取所有非打击乐器的音符
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)

        # 排序音符（按开始时间）
        notes.sort(key=lambda x: x.start)

        # 如果没有音符，返回空特征
        if not notes:
            return {
                'beat_stability': 0,
                'tempo_mean': 0,
                'tempo_std': 0,
                'rhythm_regularity': 0,
                'timing_accuracy': 0
            }

        # 计算音符间间隔(IOI - Inter-Onset Interval)
        intervals = [notes[i + 1].start - notes[i].start for i in range(len(notes) - 1)]

        # 节拍稳定性（计算IOI的变异系数 - 标准差/平均值）
        if len(intervals) > 1 and np.mean(intervals) > 0:
            beat_stability = 1.0 - min(1.0, np.std(intervals) / np.mean(intervals))
        else:
            beat_stability = 0.0

        # 获取速度变化
        tempo_changes = midi_data.get_tempo_changes()
        tempi = tempo_changes[1] if len(tempo_changes) > 1 else [120]  # 默认120BPM

        # 节奏规律性 - 基于相邻IOI的相似性
        rhythm_regularity = 0.0
        if len(intervals) > 2:
            diffs = [abs(intervals[i] - intervals[i - 1]) for i in range(1, len(intervals))]
            rhythm_regularity = 1.0 - min(1.0, np.mean(diffs) / np.mean(intervals))

        # 将特征添加到字典
        timing_features = {
            'beat_stability': beat_stability,
            'tempo_mean': np.mean(tempi),
            'tempo_std': np.std(tempi),
            'rhythm_regularity': rhythm_regularity,
            'timing_accuracy': 0.5  # 默认值，实际应与乐谱比较
        }

        return timing_features

    def extract_articulation_features(self, midi_data):
        """
        提取演奏连贯性特征（低层次）

        Args:
            midi_data: pretty_midi.PrettyMIDI对象

        Returns:
            articulation_features: 演奏连贯性特征字典
        """
        # 初始化特征字典
        articulation_features = {}

        # 如果没有有效数据，返回空特征
        if midi_data is None or len(midi_data.instruments) == 0:
            return {
                'note_length_ratio': 0,
                'legato_ratio': 0,
                'staccato_ratio': 0,
                'articulation_variability': 0
            }

        # 获取所有非打击乐器的音符
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)

        # 如果没有音符，返回空特征
        if not notes:
            return {
                'note_length_ratio': 0,
                'legato_ratio': 0,
                'staccato_ratio': 0,
                'articulation_variability': 0
            }

        # 计算音符的理论长度和实际长度
        note_length_ratios = []
        legato_count = 0
        staccato_count = 0

        for i in range(len(notes) - 1):
            current_note = notes[i]
            next_note = notes[i + 1]

            # 计算音符持续时间与间隔的比率
            theoretical_length = next_note.start - current_note.start
            actual_length = current_note.end - current_note.start

            if theoretical_length > 0:
                ratio = actual_length / theoretical_length
                note_length_ratios.append(ratio)

                # 判断连奏(legato)和断奏(staccato)
                if ratio > 0.9:  # 音符几乎连接
                    legato_count += 1
                elif ratio < 0.5:  # 音符明显断开
                    staccato_count += 1

        # 计算特征
        if note_length_ratios:
            avg_note_length_ratio = np.mean(note_length_ratios)
            articulation_variability = np.std(note_length_ratios)
        else:
            avg_note_length_ratio = 0.5  # 默认值
            articulation_variability = 0

        total_notes = len(notes)
        legato_ratio = legato_count / total_notes if total_notes > 0 else 0
        staccato_ratio = staccato_count / total_notes if total_notes > 0 else 0

        # 将特征添加到字典
        articulation_features = {
            'note_length_ratio': avg_note_length_ratio,
            'legato_ratio': legato_ratio,
            'staccato_ratio': staccato_ratio,
            'articulation_variability': articulation_variability
        }

        return articulation_features

    def extract_pedal_features(self, midi_data):
        """
        提取踏板使用特征（中低层次）

        Args:
            midi_data: pretty_midi.PrettyMIDI对象

        Returns:
            pedal_features: 踏板使用特征字典
        """
        # 初始化特征字典
        pedal_features = {}

        # 如果没有有效数据，返回空特征
        if midi_data is None or len(midi_data.instruments) == 0:
            return {
                'sustain_pedal_usage': 0,
                'pedal_density': 0,
                'pedal_duration_ratio': 0,
                'pedal_timing': 0
            }

        # 获取延音踏板控制变化
        sustain_pedal = None
        for instrument in midi_data.instruments:
            for control in instrument.control_changes:
                # MIDI控制变化64是延音踏板
                if control.number == 64:
                    if sustain_pedal is None:
                        sustain_pedal = []
                    sustain_pedal.append(control)

        # 如果没有踏板数据，返回默认值
        if sustain_pedal is None or len(sustain_pedal) == 0:
            return {
                'sustain_pedal_usage': 0,
                'pedal_density': 0,
                'pedal_duration_ratio': 0,
                'pedal_timing': 0
            }

        # 计算踏板使用密度
        duration = midi_data.get_end_time()
        pedal_density = len(sustain_pedal) / duration if duration > 0 else 0

        # 计算踏板使用时长比例
        pedal_on_time = 0
        pedal_state = False
        last_time = 0

        # 按时间排序
        sustain_pedal.sort(key=lambda x: x.time)

        for pedal_event in sustain_pedal:
            if pedal_state and pedal_event.time > last_time:
                pedal_on_time += pedal_event.time - last_time

            # MIDI踏板值 >= 64表示踏板按下
            pedal_state = pedal_event.value >= 64
            last_time = pedal_event.time

        # 最后一个状态如果是踏板按下，计算到结束
        if pedal_state:
            pedal_on_time += duration - last_time

        pedal_duration_ratio = pedal_on_time / duration if duration > 0 else 0

        # 将特征添加到字典
        pedal_features = {
            'sustain_pedal_usage': len(sustain_pedal) > 0,
            'pedal_density': pedal_density,
            'pedal_duration_ratio': pedal_duration_ratio,
            'pedal_timing': 0.5  # 默认值，实际需要更复杂的分析
        }

        return pedal_features

    def extract_dynamics_features(self, midi_data):
        """
        提取力度特征（中高层次）

        Args:
            midi_data: pretty_midi.PrettyMIDI对象

        Returns:
            dynamics_features: 力度特征字典
        """
        # 初始化特征字典
        dynamics_features = {}

        # 如果没有有效数据，返回空特征
        if midi_data is None or len(midi_data.instruments) == 0:
            return {
                'dynamic_range': 0,
                'dynamic_variability': 0,
                'velocity_mean': 0,
                'velocity_std': 0,
                'velocity_changes': 0
            }

        # 获取所有非打击乐器的音符
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                notes.extend(instrument.notes)

        # 如果没有音符，返回空特征
        if not notes:
            return {
                'dynamic_range': 0,
                'dynamic_variability': 0,
                'velocity_mean': 0,
                'velocity_std': 0,
                'velocity_changes': 0
            }

        # 提取力度(velocity)值
        velocities = [note.velocity for note in notes]

        # 计算力度特征
        vel_mean = np.mean(velocities)
        vel_std = np.std(velocities)
        vel_range = np.max(velocities) - np.min(velocities)

        # 计算力度变化
        vel_changes = [abs(velocities[i] - velocities[i - 1]) for i in range(1, len(velocities))]
        avg_vel_change = np.mean(vel_changes) if vel_changes else 0

        # 将特征添加到字典
        dynamics_features = {
            'dynamic_range': vel_range / 127.0,  # 归一化到0-1
            'dynamic_variability': vel_std / 127.0 if vel_std > 0 else 0,
            'velocity_mean': vel_mean / 127.0,
            'velocity_std': vel_std / 127.0,
            'velocity_changes': avg_vel_change / 127.0
        }

        return dynamics_features

    def extract_timbre_features(self, audio_data, sr=44100):
        """
        提取音色特征（中低层次） - 需要音频数据

        Args:
            audio_data: 音频数据
            sr: 采样率

        Returns:
            timbre_features: 音色特征字典
        """
        # 初始化特征字典
        timbre_features = {}

        # 如果没有有效数据，返回空特征
        if audio_data is None or len(audio_data) == 0:
            return {
                'spectral_centroid_mean': 0,
                'spectral_bandwidth_mean': 0,
                'spectral_contrast_mean': 0,
                'spectral_rolloff_mean': 0,
                'mfcc_mean': [0] * 13  # 13个MFCC系数
            }

        # 提取音色特征
        try:
            # 计算频谱中心
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]

            # 计算频谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]

            # 计算频谱对比度
            spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

            # 计算频谱滚降
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]

            # 计算梅尔频率倒谱系数(MFCC)
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)

            # 将特征添加到字典
            timbre_features = {
                'spectral_centroid_mean': np.mean(spectral_centroid),
                'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
                'spectral_contrast_mean': np.mean(spectral_contrast_mean),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'mfcc_mean': mfcc_mean.tolist()
            }

        except Exception as e:
            print(f"提取音色特征时出错: {e}")
            return {
                'spectral_centroid_mean': 0,
                'spectral_bandwidth_mean': 0,
                'spectral_contrast_mean': 0,
                'spectral_rolloff_mean': 0,
                'mfcc_mean': [0] * 13
            }

        return timbre_features

    def extract_high_level_features(self, midi_data, audio_data=None, sr=44100):
        """
        提取高层次特征（音乐构建、情感、演绎）
        这些特征通常综合低层次特征计算得到

        Args:
            midi_data: pretty_midi.PrettyMIDI对象
            audio_data: 音频数据（可选）
            sr: 采样率

        Returns:
            high_level_features: 高层次特征字典
        """
        # 提取低层次特征
        timing_features = self.extract_timing_features(midi_data)
        articulation_features = self.extract_articulation_features(midi_data)
        pedal_features = self.extract_pedal_features(midi_data)
        dynamics_features = self.extract_dynamics_features(midi_data)

        # 提取音色特征（如果有音频数据）
        timbre_features = self.extract_timbre_features(audio_data, sr) if audio_data is not None else {}

        # 初始化高层次特征字典
        high_level_features = {}

        # 1. 音乐构建特征（中高层次）
        music_making_features = {
            # 节奏灵活性 - 基于节奏规律性和速度变化
            'rhythm_flexibility': 0.5 * (1 - timing_features['rhythm_regularity']) + 0.5 * (
                timing_features['tempo_std'] / 50 if timing_features['tempo_std'] < 50 else 1),

            # 空间感 - 基于力度变化和踏板使用
            'spaciousness': 0.5 * dynamics_features['dynamic_range'] + 0.5 * pedal_features['pedal_duration_ratio'],

            # 平衡度 - 基于音域和力度分布
            'balance': 0.5,  # 需要更复杂的分析

            # 戏剧性 - 基于力度变化和力度范围
            'dramatic': 0.7 * dynamics_features['dynamic_range'] + 0.3 * dynamics_features['velocity_changes']
        }

        # 2. 情感特征（高层次）
        emotion_features = {
            # 愉悦/悲伤 - 基于速度、力度和音色
            'pleasant_sad': 0.4 * (1 - dynamics_features['velocity_mean']) + 0.3 * (
                        1 - timing_features['tempo_mean'] / 120) + 0.3 * articulation_features['legato_ratio'],

            # 能量 - 基于速度、力度和节奏稳定性
            'energy': 0.4 * dynamics_features['velocity_mean'] + 0.4 * (timing_features['tempo_mean'] / 120) + 0.2 *
                      timing_features['beat_stability'],

            # 诚实/想象力 - 这是比较主观的
            'honest_imaginative': 0.5  # 默认中间值
        }

        # 3. 演绎特征（高层次）
        interpretation_features = {
            # 令人信服度 - 这是一个综合的主观评估
            'convincing': 0.3 * timing_features['beat_stability'] + 0.2 * dynamics_features['dynamic_range'] +
                          0.2 * articulation_features['articulation_variability'] + 0.3 * music_making_features[
                              'balance']
        }

        # 合并所有高层次特征
        high_level_features.update(music_making_features)
        high_level_features.update(emotion_features)
        high_level_features.update(interpretation_features)

        return high_level_features

    def extract_all_features(self, midi_data, audio_data=None, sr=44100):
        """
        提取所有特征

        Args:
            midi_data: pretty_midi.PrettyMIDI对象
            audio_data: 音频数据（可选）
            sr: 采样率

        Returns:
            all_features: 所有特征字典
        """
        # 提取各级别特征
        timing_features = self.extract_timing_features(midi_data)
        articulation_features = self.extract_articulation_features(midi_data)
        pedal_features = self.extract_pedal_features(midi_data)
        dynamics_features = self.extract_dynamics_features(midi_data)

        # 如果有音频数据，提取音色特征
        timbre_features = {}
        if audio_data is not None:
            timbre_features = self.extract_timbre_features(audio_data, sr)

        # 提取高层次特征
        high_level_features = self.extract_high_level_features(midi_data, audio_data, sr)

        # 合并所有特征
        all_features = {}
        all_features.update(timing_features)
        all_features.update(articulation_features)
        all_features.update(pedal_features)
        all_features.update(dynamics_features)
        all_features.update(timbre_features)
        all_features.update(high_level_features)

        return all_features

    def create_feature_vector(self, features_dict):
        """
        将特征字典转换为特征向量（用于机器学习）

        Args:
            features_dict: 特征字典

        Returns:
            feature_vector: 特征向量
        """
        # 创建特征向量（扁平化嵌套结构）
        feature_vector = []
        for key, value in features_dict.items():
            if isinstance(value, (list, np.ndarray)):
                feature_vector.extend(value)
            else:
                feature_vector.append(value)

        return np.array(feature_vector)


# 使用示例
if __name__ == "__main__":
    import pretty_midi

    # 创建特征提取器
    extractor = PianoFeatureExtractor()

    # 加载MIDI文件
    midi_file = "path/to/your/midi_file.mid"
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_file)

        # 提取所有特征
        features = extractor.extract_all_features(midi_data)

        # 打印特征
        print("提取的特征:")
        for key, value in features.items():
            print(f"{key}: {value}")

        # 创建特征向量
        feature_vector = extractor.create_feature_vector(features)
        print(f"\n特征向量形状: {feature_vector.shape}")

    except Exception as e:
        print(f"处理MIDI文件时出错: {e}")