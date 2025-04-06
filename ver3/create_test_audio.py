"""
创建测试用的钢琴音频文件
"""

import os
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from scipy.io import wavfile
import matplotlib.pyplot as plt

def generate_piano_note(freq, duration=1.0, sample_rate=32000, amp=0.5):
    """生成钢琴音符"""
    # 创建时间数组
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # 生成基本正弦波
    note = amp * np.sin(2 * np.pi * freq * t)
    
    # 添加谐波以模拟钢琴音色
    for i in range(2, 6):
        harmonic = (amp / i) * np.sin(2 * np.pi * (freq * i) * t)
        note += harmonic
    
    # 添加衰减包络
    envelope = np.exp(-3 * t)
    note = note * envelope
    
    return note

def generate_piano_melody(sample_rate=32000):
    """生成钢琴旋律"""
    # C大调音阶的频率
    c_major_scale = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23,
        'G4': 392.00, 'A4': 440.00, 'B4': 493.88, 'C5': 523.25
    }
    
    # 旋律序列
    melody = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4', 'D4', 'F4', 'A4', 'F4', 'D4',
              'E4', 'G4', 'C5', 'G4', 'E4', 'C4']
    
    # 每个音符的持续时间
    durations = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 
                 0.5, 0.5, 0.5, 0.5, 1.0,
                 0.5, 0.5, 0.5, 0.5, 0.5, 1.0]
    
    # 创建空的音频数组
    total_duration = sum(durations)
    audio = np.zeros(int(sample_rate * total_duration))
    
    # 生成旋律
    current_time = 0
    for note, duration in zip(melody, durations):
        freq = c_major_scale[note]
        note_audio = generate_piano_note(freq, duration, sample_rate)
        
        # 将音符添加到总音频中
        start_idx = int(current_time * sample_rate)
        end_idx = start_idx + len(note_audio)
        
        # 确保不超出数组边界
        if end_idx <= len(audio):
            audio[start_idx:end_idx] += note_audio
        
        current_time += duration
    
    # 归一化
    audio = audio / np.max(np.abs(audio))
    
    return audio, sample_rate

def generate_complex_piano_piece(sample_rate=32000, duration=10.0):
    """生成更复杂的钢琴片段，包含和弦"""
    # 设置常见和弦的频率（C大调）
    chords = {
        'C': [261.63, 329.63, 392.00],  # C 大三和弦 (C, E, G)
        'F': [349.23, 440.00, 523.25],  # F 大三和弦 (F, A, C)
        'G': [392.00, 493.88, 587.33],  # G 大三和弦 (G, B, D)
        'Am': [220.00, 261.63, 329.63]   # A 小三和弦 (A, C, E)
    }
    
    # 和弦进行 (每个和弦持续时间)
    progression = [('C', 2.0), ('F', 2.0), ('G', 2.0), ('C', 2.0), ('Am', 2.0)]
    
    # 创建空的音频数组
    total_samples = int(sample_rate * duration)
    audio = np.zeros(total_samples)
    
    # 当前位置
    current_sample = 0
    
    # 为每个和弦添加音频
    for chord_name, chord_duration in progression:
        if current_sample >= total_samples:
            break
            
        # 确定此和弦的采样数
        chord_samples = int(sample_rate * chord_duration)
        
        # 为此和弦创建所有音符
        chord_audio = np.zeros(chord_samples)
        
        # 添加和弦中的每个音符
        for freq in chords[chord_name]:
            note = generate_piano_note(freq, chord_duration, sample_rate, amp=0.3)
            chord_audio += note
        
        # 添加一些简单的旋律装饰
        melody_notes = np.random.choice(chords[chord_name], 8)
        for i, freq in enumerate(melody_notes):
            # 为每个装饰音符创建一个短音符
            start = int(i * chord_samples / 8)
            length = int(chord_samples / 16)
            if start + length <= chord_samples:
                melody_note = generate_piano_note(freq * 2, chord_duration/16, sample_rate, amp=0.2)
                chord_audio[start:start+len(melody_note)] += melody_note
        
        # 添加到主音频
        end_sample = min(current_sample + chord_samples, total_samples)
        audio[current_sample:end_sample] += chord_audio[:end_sample-current_sample]
        
        current_sample += chord_samples
    
    # 归一化
    audio = audio / np.max(np.abs(audio))
    
    return audio, sample_rate

def save_audio(audio, sample_rate, filename):
    """保存音频文件"""
    # 确保音频值在-1到1之间
    audio = np.clip(audio, -1, 1)
    
    # 转换为16位整数
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # 保存为WAV文件
    wavfile.write(filename, sample_rate, audio_int16)
    print(f"音频已保存至: {filename}")
    
    return filename

def plot_waveform(audio, sample_rate, filename=None):
    """绘制音频波形"""
    plt.figure(figsize=(10, 4))
    time = np.arange(0, len(audio)) / sample_rate
    plt.plot(time, audio)
    plt.title("Piano Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"波形图已保存至: {filename}")
    else:
        plt.show()

def main():
    """主函数"""
    # 创建输出目录
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成简单钢琴旋律
    print("生成简单钢琴旋律...")
    simple_audio, sample_rate = generate_piano_melody()
    simple_file = os.path.join(output_dir, "simple_piano.wav")
    save_audio(simple_audio, sample_rate, simple_file)
    plot_waveform(simple_audio, sample_rate, os.path.join(output_dir, "simple_piano_waveform.png"))
    
    # 生成复杂钢琴片段
    print("生成复杂钢琴片段...")
    complex_audio, sample_rate = generate_complex_piano_piece(duration=15.0)
    complex_file = os.path.join(output_dir, "complex_piano.wav")
    save_audio(complex_audio, sample_rate, complex_file)
    plot_waveform(complex_audio, sample_rate, os.path.join(output_dir, "complex_piano_waveform.png"))
    
    print(f"测试音频生成完成，文件保存在 {output_dir} 目录中")
    return {
        "simple_piano": simple_file,
        "complex_piano": complex_file
    }

if __name__ == "__main__":
    main() 