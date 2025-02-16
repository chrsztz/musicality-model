# src/feature_extraction.py

import numpy as np
import pretty_midi
import librosa


class FeatureExtractor:
    def __init__(self):
        pass

    def extract_features(self, midi: pretty_midi.PrettyMIDI):
        features = {}
        # 1. 节奏特征
        tempo = midi.estimate_tempo()
        features['tempo'] = tempo

        # 2. 音高特征
        pitches = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitches.append(note.pitch)
        avg_pitch = np.mean(pitches) if pitches else 0
        features['avg_pitch'] = avg_pitch

        # 3. 力度特征
        velocities = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    velocities.append(note.velocity)
        avg_velocity = np.mean(velocities) if velocities else 0
        features['avg_velocity'] = avg_velocity

        # 4. 动态特征（如pedal使用）
        pedaling = self._extract_pedal_features(midi)
        features.update(pedaling)

        # 5. 更多特征根据 PercePiano 数据集的19个感知特征进行扩展
        # 示例：
        # features['articulation'] = ...
        # features['timbre'] = ...

        return features

    def _extract_pedal_features(self, midi: pretty_midi.PrettyMIDI):
        # 示例提取pedal使用情况
        # 具体根据数据集和需求调整
        pedal_times = []
        for control_change in midi.get_piano_roll().reshape(-1):
            # 假设CC64为pedal，具体需要确认
            pass
        # 返回一个示例
        return {'pedal_usage': 0}  # 替换为实际计算值

    def vectorize_features(self, features: dict):
        # 将特征字典转换为固定长度的向量
        # 按照固定顺序排列特征
        feature_order = sorted(features.keys())
        return np.array([features[key] for key in feature_order])
