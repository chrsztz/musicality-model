import os
import numpy as np
import pandas as pd
import pretty_midi
import librosa
from tqdm import tqdm


class MIDIDataProcessor:
    """MIDI数据处理类，负责加载MIDI文件并提取特征"""

    def __init__(self, data_dir='../PercePiano/virtuoso/data/all_2rounds', labels_path='../PercePiano/labels/total_2rounds.csv'):
        """
        初始化数据处理器

        Args:
            data_dir: MIDI文件目录
            labels_path: 标签文件路径
        """
        self.data_dir = data_dir
        self.labels_path = labels_path

    def load_labels(self):
        """加载CSV标签文件"""
        try:
            df = pd.read_csv(self.labels_path)
            print(f"成功加载标签数据，包含{len(df)}条记录")
            print(f"标签列: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"加载标签数据失败: {e}")
            return None

    def load_midi(self, midi_path):
        """
        加载MIDI文件

        Args:
            midi_path: MIDI文件路径

        Returns:
            pretty_midi.PrettyMIDI对象或None（如果加载失败）
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            return midi_data
        except Exception as e:
            print(f"加载MIDI文件 {midi_path} 失败: {e}")
            return None

    def extract_pitch_contour(self, midi_data, fs=100, normalize=True):
        """
        从MIDI数据中提取音高轮廓

        Args:
            midi_data: pretty_midi.PrettyMIDI对象
            fs: 每秒采样点数
            normalize: 是否归一化音高（0-1之间）

        Returns:
            音高轮廓数组
        """
        if midi_data is None:
            return None

        # 获取总时长
        duration = midi_data.get_end_time()
        times = np.arange(0, duration, 1 / fs)

        # 提取音高轮廓
        pitches = []
        for time in times:
            notes = midi_data.get_active_notes(time)
            if notes:
                # 如果存在多个音符，取最高音
                pitch = max(note.pitch for note in notes)
                pitches.append(pitch)
            else:
                # 休止符用0表示
                pitches.append(0)

        # 归一化处理
        if normalize and pitches:
            # 乐器音域通常为C2(36)到C8(108)
            min_pitch, max_pitch = 36, 108
            pitches = np.array(pitches)
            # 将休止符(0)保持为0，其他音符归一化到0-1
            mask = pitches > 0
            if np.any(mask):
                pitches[mask] = (pitches[mask] - min_pitch) / (max_pitch - min_pitch)

        return np.array(pitches)

    def extract_dynamics(self, midi_data, fs=100):
        """
        从MIDI数据中提取力度变化

        Args:
            midi_data: pretty_midi.PrettyMIDI对象
            fs: 每秒采样点数

        Returns:
            力度变化数组（归一化到0-1）
        """
        if midi_data is None:
            return None

        # 获取总时长
        duration = midi_data.get_end_time()
        times = np.arange(0, duration, 1 / fs)

        # 提取力度变化
        velocities = []
        for time in times:
            notes = midi_data.get_active_notes(time)
            if notes:
                # 计算平均力度
                vel = np.mean([note.velocity for note in notes])
                velocities.append(vel)
            else:
                # 休止符用0表示
                velocities.append(0)

        # 归一化到0-1
        velocities = np.array(velocities)
        if np.max(velocities) > 0:
            velocities = velocities / 127.0  # MIDI力度范围0-127

        return velocities

    def extract_rhythm_features(self, midi_data):
        """
        提取节奏特征

        Args:
            midi_data: pretty_midi.PrettyMIDI对象

        Returns:
            节奏特征字典
        """
        if midi_data is None:
            return None

        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:  # 排除打击乐器
                notes.extend(instrument.notes)

        if not notes:
            return {'avg_ioi': 0, 'ioi_std': 0, 'tempo_mean': 0, 'tempo_std': 0}

        # 按开始时间排序
        notes.sort(key=lambda x: x.start)

        # 计算音符间隔(IOI - Inter-Onset Interval)
        note_intervals = [notes[i + 1].start - notes[i].start for i in range(len(notes) - 1)]

        # 获取速度变化
        tempo_changes = midi_data.get_tempo_changes()
        tempi = tempo_changes[1] if len(tempo_changes) > 1 else [120]  # 默认120BPM

        return {
            'avg_ioi': np.mean(note_intervals) if note_intervals else 0,
            'ioi_std': np.std(note_intervals) if note_intervals else 0,
            'tempo_mean': np.mean(tempi),
            'tempo_std': np.std(tempi)
        }

    def extract_all_features(self, midi_path):
        """
        从MIDI文件中提取所有特征

        Args:
            midi_path: MIDI文件路径

        Returns:
            特征字典
        """
        midi_data = self.load_midi(midi_path)
        if midi_data is None:
            return None

        # 提取基本特征
        pitch_contour = self.extract_pitch_contour(midi_data)
        dynamics = self.extract_dynamics(midi_data)
        rhythm_features = self.extract_rhythm_features(midi_data)

        # 提取高级特征
        pitch_range = 0
        if len(midi_data.instruments) > 0 and len(midi_data.instruments[0].notes) > 0:
            pitches = [note.pitch for instr in midi_data.instruments
                       for note in instr.notes if not instr.is_drum]
            if pitches:
                pitch_range = max(pitches) - min(pitches)

        return {
            'pitch_contour': pitch_contour,
            'dynamics': dynamics,
            'rhythm_features': rhythm_features,
            'pitch_range': pitch_range,
            'duration': midi_data.get_end_time(),
            'num_notes': sum(len(instr.notes) for instr in midi_data.instruments)
        }

    def process_dataset(self, limit=None):
        """
        处理整个数据集

        Args:
            limit: 限制处理的文件数量（用于测试）

        Returns:
            特征数据和标签的列表
        """
        labels_df = self.load_labels()
        if labels_df is None:
            return None

        features_list = []
        labels_list = []
        valid_files = []

        # 限制处理的文件数量
        if limit:
            labels_df = labels_df.head(limit)

        for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="处理MIDI文件"):
            # 构建文件路径（可能需要根据实际数据调整）
            file_id = row.get('file_id', idx)
            midi_path = os.path.join(self.data_dir, f"{file_id}.mid")

            # 检查文件是否存在
            if not os.path.exists(midi_path):
                print(f"警告: 文件 {midi_path} 不存在，跳过")
                continue

            # 提取特征
            features = self.extract_all_features(midi_path)
            if features is None:
                print(f"警告: 从 {midi_path} 提取特征失败，跳过")
                continue

            # 提取标签
            labels = row.to_dict()

            features_list.append(features)
            labels_list.append(labels)
            valid_files.append(midi_path)

        print(f"成功处理 {len(valid_files)} 个有效文件")
        return features_list, labels_list, valid_files

    def synthesize_audio(self, midi_data, sr=44100):
        """
        将MIDI数据合成为音频

        Args:
            midi_data: pretty_midi.PrettyMIDI对象
            sr: 采样率

        Returns:
            音频数据数组
        """
        if midi_data is None:
            return None

        try:
            audio_data = midi_data.synthesize(fs=sr)
            return audio_data
        except Exception as e:
            print(f"MIDI合成音频失败: {e}")
            return None

    def extract_audio_features(self, audio_data, sr=44100):
        """
        从合成的音频中提取音频特征

        Args:
            audio_data: 音频数据数组
            sr: 采样率

        Returns:
            音频特征字典
        """
        if audio_data is None:
            return None

        # 提取梅尔频谱
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )

        # 转换为分贝单位
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 提取MFCC
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=13)

        # 计算统计特征
        features = {
            'mel_mean': np.mean(mel_spec_db, axis=1),
            'mel_std': np.std(mel_spec_db, axis=1),
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]),
            'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0])
        }

        return features


# 使用示例
if __name__ == "__main__":
    processor = MIDIDataProcessor()
    # 测试处理少量文件
    features, labels, files = processor.process_dataset(limit=5)

    if features:
        # 打印第一个文件的特征
        print("\n第一个文件的特征:")
        first_features = features[0]
        for key, value in first_features.items():
            if isinstance(value, np.ndarray):
                print(f"{key}: 数组形状 {value.shape}")
            else:
                print(f"{key}: {value}")

        # 打印第一个文件的标签
        print("\n第一个文件的标签:")
        for key, value in labels[0].items():
            print(f"{key}: {value}")