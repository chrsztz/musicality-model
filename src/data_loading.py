# src/data_loading.py

import os
import pandas as pd
import pretty_midi
from feature_extraction import FeatureExtractor
import torch
import numpy as np
from torch.utils.data import Dataset

class PianoDataset(Dataset):
    def __init__(self, data, feature_extractor):
        self.data = data
        self.feature_extractor = feature_extractor
        self.feature_order = sorted(next(iter(data))[0].keys())  # 获取特征顺序

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, labels = self.data[idx]
        feature_vector = self.feature_extractor.vectorize_features(features)
        label_vector = np.array(list(labels.values()))
        return torch.tensor(feature_vector, dtype=torch.float32), torch.tensor(label_vector, dtype=torch.float32)

class PianoPerformanceDataset:
    def __init__(self, midi_dir, label_csv):
        self.midi_dir = midi_dir
        self.labels = pd.read_csv(label_csv)
        self.feature_extractor = FeatureExtractor()
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for idx, row in self.labels.iterrows():
            midi_file = row['midi_file']  # 根据实际列名调整
            midi_path = os.path.join(self.midi_dir, midi_file)
            if os.path.exists(midi_path):
                try:
                    midi = pretty_midi.PrettyMIDI(midi_path)
                    features = self.feature_extractor.extract_features(midi)
                    labels = self._extract_labels(row)
                    data.append((features, labels))
                except Exception as e:
                    print(f"Error processing {midi_path}: {e}")
            else:
                print(f"MIDI file not found: {midi_path}")
        return data

    def _extract_labels(self, row):
        # 假设有19个感知特征，每个特征有一个分数
        # 根据实际列名和结构调整
        labels = row.drop('midi_file').to_dict()
        return labels

    def get_data(self):
        return self.data


if __name__ == "__main__":
    midi_dir = 'data/all_2rounds'
    label_csv = 'data/labels/total_2rounds.csv'
    dataset = PianoPerformanceDataset(midi_dir, label_csv)
    print(f"Loaded {len(dataset.get_data())} samples.")
