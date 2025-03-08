import numpy as np
import sounddevice as sd
import torch
import threading
import queue
import time
import librosa
import pretty_midi
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
import os

# 导入自定义模块
from feature_extraction import PianoFeatureExtractor

warnings.filterwarnings('ignore')


class RealTimePianoAssessment:
    """
    实时钢琴演奏评估系统
    可以捕获音频输入，提取特征，并使用预训练模型进行实时评估
    """

    def __init__(self, model, model_type='pc_cnn', device=None,
                 sample_rate=44100, block_duration=2.0, overlap=0.5,
                 task_names=None, multi_task=False):
        """
        初始化实时评估系统

        Args:
            model: 预训练模型
            model_type: 模型类型 ('pc_cnn', 'mel_crnn', 'hybrid')
            device: 推理设备 (CPU或GPU)
            sample_rate: 采样率
            block_duration: 音频块持续时间（秒）
            overlap: 音频块重叠比例
            task_names: 任务名称列表（多任务模型）
            multi_task: 是否为多任务模型
        """
        self.model = model
        self.model_type = model_type
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.overlap = overlap
        self.task_names = task_names or ['整体评分']
        self.multi_task = multi_task

        # 特征提取器
        self.feature_extractor = PianoFeatureExtractor()

        # 将模型移动到指定设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()

        # 计算音频块大小和步长
        self.block_size = int(self.block_duration * self.sample_rate)
        self.hop_size = int(self.block_size * (1 - self.overlap))

        # 音频流和处理线程
        self.audio_queue = queue.Queue()
        self.audio_buffer = np.zeros(0)
        self.is_recording = False
        self.stream = None
        self.processing_thread = None

        # 评估结果
        self.current_scores = {task: 0.0 for task in self.task_names}
        self.scores_history = {task: [] for task in self.task_names}
        self.timestamp_history = []

        # 音符检测和MIDI转换参数
        self.pitch_threshold = 0.7  # 音符检测阈值
        self.min_note_duration = 0.1  # 最小音符持续时间（秒）
        self.detected_notes = []  # 检测到的音符列表

        # 可视化参数
        self.fig = None
        self.axes = None
        self.animation = None

        print(f"实时钢琴演奏评估系统初始化完成，使用{self.device}进行推理")

    def audio_callback(self, indata, frames, time, status):
        """
        音频流回调函数，接收实时音频输入

        Args:
            indata: 输入音频数据
            frames: 帧数
            time: 时间信息
            status: 状态标志
        """
        if status:
            print(f"音频流状态: {status}")

        # 仅使用第一个通道（如果有多个）
        audio_data = indata[:, 0].copy()
        self.audio_queue.put(audio_data)

    def process_audio(self):
        """
        处理实时音频流的线程函数
        提取特征并进行评估
        """
        try:
            start_time = time.time()

            while self.is_recording:
                # 从队列获取音频数据
                try:
                    audio_data = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 添加到缓冲区
                self.audio_buffer = np.concatenate((self.audio_buffer, audio_data))

                # 当缓冲区足够大时进行处理
                if len(self.audio_buffer) >= self.block_size:
                    # 提取当前块
                    current_block = self.audio_buffer[:self.block_size]

                    # 更新缓冲区，移除除重叠部分外的数据
                    self.audio_buffer = self.audio_buffer[self.hop_size:]

                    # 处理当前块
                    self._process_audio_block(current_block, time.time() - start_time)

        except Exception as e:
            print(f"音频处理线程出错: {e}")

        print("音频处理线程已停止")

    def _process_audio_block(self, audio_block, timestamp):
        """
        处理单个音频块

        Args:
            audio_block: 音频数据块
            timestamp: 时间戳
        """
        # 提取特征
        features = self._extract_features(audio_block)
        if features is None:
            return

        # 进行推理
        with torch.no_grad():
            # 准备输入数据
            if self.model_type == 'pc_cnn':
                # 音高轮廓输入
                pc_input = torch.tensor(features['pitch_contour'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                pc_input = pc_input.to(self.device)
                outputs = self.model(pc_input)

            elif self.model_type == 'mel_crnn':
                # 梅尔频谱输入
                mel_input = torch.tensor(features['mel_spectrogram'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_input = mel_input.to(self.device)
                outputs = self.model(mel_input)

            else:  # hybrid
                # 混合输入
                pc_input = torch.tensor(features['pitch_contour'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_input = torch.tensor(features['mel_spectrogram'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                pc_input = pc_input.to(self.device)
                mel_input = mel_input.to(self.device)
                outputs = self.model(pc_input, mel_input)

            # 处理输出
            outputs = outputs.cpu().numpy()

            # 更新当前评分
            if self.multi_task:
                for i, task in enumerate(self.task_names):
                    self.current_scores[task] = float(outputs[0, i])
                    self.scores_history[task].append(float(outputs[0, i]))
            else:
                # 单任务输出 - 仅更新第一个任务
                self.current_scores[self.task_names[0]] = float(outputs[0, 0])
                self.scores_history[self.task_names[0]].append(float(outputs[0, 0]))

            # 添加时间戳
            self.timestamp_history.append(timestamp)

            # 打印当前评分
            self._print_current_scores()

    def _extract_features(self, audio_block):
        """
        从音频块中提取特征

        Args:
            audio_block: 音频数据块

        Returns:
            features: 特征字典
        """
        try:
            # 使用 librosa 进行音高检测
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio_block,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate
            )

            # 创建音高轮廓 (MIDI音高)
            pitch_contour = np.zeros_like(f0)
            voiced_indices = np.where(voiced_flag)[0]
            if len(voiced_indices) > 0:
                # 频率到MIDI音高的转换 (MIDI = 69 + 12*log2(f/440))
                pitch_contour[voiced_indices] = 69 + 12 * np.log2(f0[voiced_indices] / 440.0)

                # 归一化到0-1范围 (C2 = 36 到 C7 = 96)
                pitch_contour[voiced_indices] = (pitch_contour[voiced_indices] - 36) / 60

                # 限制在0-1范围内
                pitch_contour[voiced_indices] = np.clip(pitch_contour[voiced_indices], 0, 1)

            # 检测音符（用于可视化和MIDI转换）
            self._detect_notes(f0, voiced_flag)

            # 提取梅尔频谱
            mel_spec = librosa.feature.melspectrogram(
                y=audio_block,
                sr=self.sample_rate,
                n_fft=2048,
                hop_length=512,
                n_mels=128
            )

            # 转换为分贝单位
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # 创建特征字典
            features = {
                'pitch_contour': pitch_contour,
                'mel_spectrogram': mel_spec_db
            }

            return features

        except Exception as e:
            print(f"特征提取出错: {e}")
            return None

    def _detect_notes(self, f0, voiced_flag):
        """
        检测音符并更新检测到的音符列表

        Args:
            f0: 基频估计
            voiced_flag: 有声标志
        """
        current_note = None

        for i, (freq, voiced) in enumerate(zip(f0, voiced_flag)):
            time_point = i / len(f0) * self.block_duration

            if voiced and freq > 0:
                # 频率到MIDI音高的转换
                midi_pitch = int(round(69 + 12 * np.log2(freq / 440.0)))

                if current_note is None:
                    # 开始新音符
                    current_note = {
                        'pitch': midi_pitch,
                        'start': time_point,
                        'end': time_point
                    }
                elif midi_pitch == current_note['pitch']:
                    # 延长当前音符
                    current_note['end'] = time_point
                else:
                    # 结束当前音符并开始新音符
                    if current_note['end'] - current_note['start'] >= self.min_note_duration:
                        self.detected_notes.append(current_note)

                    current_note = {
                        'pitch': midi_pitch,
                        'start': time_point,
                        'end': time_point
                    }
            elif current_note is not None:
                # 无声 - 结束当前音符
                if time_point - current_note['start'] >= self.min_note_duration:
                    current_note['end'] = time_point
                    self.detected_notes.append(current_note)

                current_note = None

        # 处理最后一个音符
        if current_note is not None and current_note['end'] - current_note['start'] >= self.min_note_duration:
            self.detected_notes.append(current_note)

    def _print_current_scores(self):
        """打印当前评分"""
        print("\n当前评分:")
        for task, score in self.current_scores.items():
            # 将分数映射到0-10范围
            score_10 = score * 10
            print(f"{task}: {score_10:.2f}/10")

    def start(self):
        """启动实时评估"""
        if self.is_recording:
            print("已经在录制中")
            return

        # 重置状态
        self.audio_buffer = np.zeros(0)
        self.current_scores = {task: 0.0 for task in self.task_names}
        self.scores_history = {task: [] for task in self.task_names}
        self.timestamp_history = []
        self.detected_notes = []

        # 开始录制
        self.is_recording = True

        # 启动音频流
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self.audio_callback,
            blocksize=self.hop_size
        )
        self.stream.start()

        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("实时评估已启动，开始演奏...")

    def stop(self):
        """停止实时评估"""
        if not self.is_recording:
            print("没有进行中的录制")
            return

        # 停止录制
        self.is_recording = False

        # 停止音频流
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # 等待处理线程结束
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            self.processing_thread = None

        print("实时评估已停止")

        # 显示最终评估结果
        self._print_final_results()

    def _print_final_results(self):
        """打印最终评估结果"""
        if not self.scores_history[self.task_names[0]]:
            print("没有足够的数据进行评估")
            return

        print("\n=== 最终评估结果 ===")
        for task in self.task_names:
            # 计算平均分
            avg_score = np.mean(self.scores_history[task])
            # 映射到0-10范围
            avg_score_10 = avg_score * 10
            print(f"{task}: {avg_score_10:.2f}/10")

    def visualize(self):
        """可视化评估结果"""
        if not self.scores_history[self.task_names[0]]:
            print("没有数据可供可视化")
            return

        # 创建图形
        plt.figure(figsize=(15, 10))

        # 1. 评分历史
        plt.subplot(2, 1, 1)
        for task in self.task_names:
            # 将分数映射到0-10范围
            scores_10 = [score * 10 for score in self.scores_history[task]]
            plt.plot(self.timestamp_history, scores_10, label=task)

        plt.title('评分历史')
        plt.xlabel('时间 (秒)')
        plt.ylabel('评分 (0-10)')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 10)

        # 2. 检测到的音符
        plt.subplot(2, 1, 2)
        if self.detected_notes:
            for note in self.detected_notes:
                plt.plot(
                    [note['start'], note['end']],
                    [note['pitch'], note['pitch']],
                    'b-',
                    linewidth=2
                )

        plt.title('检测到的音符')
        plt.xlabel('时间 (秒)')
        plt.ylabel('MIDI音高')
        plt.grid(True)
        plt.ylim(36, 96)  # C2到C7的MIDI音高范围

        plt.tight_layout()
        plt.savefig('assessment_visualization.png')
        plt.show()

    def start_interactive_visualization(self):
        """启动交互式可视化"""
        # 创建图形和轴
        self.fig, self.axes = plt.subplots(2, 1, figsize=(12, 8))

        # 评分历史轴
        self.score_lines = {}
        for task in self.task_names:
            self.score_lines[task], = self.axes[0].plot([], [], label=task)

        self.axes[0].set_title('实时评分')
        self.axes[0].set_xlabel('时间 (秒)')
        self.axes[0].set_ylabel('评分 (0-10)')
        self.axes[0].legend()
        self.axes[0].grid(True)
        self.axes[0].set_ylim(0, 10)
        self.axes[0].set_xlim(0, 60)  # 默认显示60秒

        # 音符可视化轴
        self.note_scatter = self.axes[1].scatter([], [], s=30, c='blue')
        self.axes[1].set_title('检测到的音符')
        self.axes[1].set_xlabel('时间 (秒)')
        self.axes[1].set_ylabel('MIDI音高')
        self.axes[1].grid(True)
        self.axes[1].set_ylim(36, 96)  # C2到C7
        self.axes[1].set_xlim(0, 60)  # 默认显示60秒

        plt.tight_layout()

        # 动画更新函数
        def update(frame):
            # 更新评分历史
            for task in self.task_names:
                if self.scores_history[task]:
                    # 映射到0-10范围
                    scores_10 = [score * 10 for score in self.scores_history[task]]
                    self.score_lines[task].set_data(self.timestamp_history, scores_10)

            # 更新x轴范围
            if self.timestamp_history:
                max_time = max(self.timestamp_history)
                if max_time > self.axes[0].get_xlim()[1]:
                    self.axes[0].set_xlim(0, max_time + 10)
                    self.axes[1].set_xlim(0, max_time + 10)

            # 更新音符可视化
            if self.detected_notes:
                note_x = []
                note_y = []
                for note in self.detected_notes:
                    # 添加音符开始点
                    note_x.append(note['start'])
                    note_y.append(note['pitch'])
                    # 添加音符结束点
                    note_x.append(note['end'])
                    note_y.append(note['pitch'])

                self.note_scatter.set_offsets(np.column_stack((note_x, note_y)))

            # 返回需要更新的artists
            artists = list(self.score_lines.values())
            artists.append(self.note_scatter)
            return artists

        # 创建动画
        self.animation = FuncAnimation(
            self.fig, update, interval=500, blit=True, cache_frame_data=False
        )

        plt.show(block=False)

    def save_midi(self, output_path='detected_notes.mid'):
        """
        将检测到的音符保存为MIDI文件

        Args:
            output_path: 输出MIDI文件路径
        """
        if not self.detected_notes:
            print("没有检测到音符可保存")
            return

        # 创建MIDI文件
        midi = pretty_midi.PrettyMIDI()

        # 创建钢琴乐器
        piano = pretty_midi.Instrument(program=0)  # 0是钢琴

        # 添加检测到的音符
        for note_data in self.detected_notes:
            note = pretty_midi.Note(
                velocity=80,  # 默认力度
                pitch=note_data['pitch'],
                start=note_data['start'],
                end=note_data['end']
            )
            piano.notes.append(note)

        # 将乐器添加到MIDI文件
        midi.instruments.append(piano)

        # 保存MIDI文件
        midi.write(output_path)
        print(f"MIDI文件已保存到 {output_path}")

    def save_audio(self, output_path='recorded_audio.wav'):
        """
        保存录制的音频

        Args:
            output_path: 输出WAV文件路径
        """
        if len(self.audio_buffer) == 0:
            print("没有录制的音频可保存")
            return

        # 保存为WAV文件
        wavfile.write(output_path, self.sample_rate, self.audio_buffer)
        print(f"音频文件已保存到 {output_path}")

    def generate_feedback(self, llm_api=None):
        """
        生成详细的文字反馈

        Args:
            llm_api: 大语言模型API调用函数（可选）

        Returns:
            feedback: 文字反馈字符串
        """
        if not self.scores_history[self.task_names[0]]:
            return "没有足够的数据生成反馈"

        # 计算平均分
        average_scores = {}
        for task in self.task_names:
            average_scores[task] = np.mean(self.scores_history[task])

        # 如果提供了LLM API，则使用它生成详细反馈
        if llm_api:
            try:
                # 准备提示
                prompt = f"""
                我是一位钢琴演奏者，刚刚完成了一段演奏，AI系统对我的演奏进行了评估，得分如下（满分10分）：

                """
                for task, score in average_scores.items():
                    prompt += f"{task}: {score * 10:.2f}/10\n"

                prompt += """
                请基于这些评分给我详细的反馈，包括：
                1. 总体评价
                2. 我的优势部分
                3. 需要改进的部分
                4. 具体的改进建议

                请用专业但易于理解的语言，像一位钢琴老师一样给予建设性的反馈。
                """

                # 调用LLM API
                feedback = llm_api(prompt)
                return feedback

            except Exception as e:
                print(f"调用LLM API出错: {e}")
                # 如果API调用失败，回退到默认反馈

        # 默认反馈生成逻辑
        feedback = "=== 演奏评估反馈 ===\n\n"

        # 总体评价
        overall_score = np.mean(list(average_scores.values())) * 10
        if overall_score >= 8.5:
            feedback += "总体评价: 您的演奏非常出色！展现了很高的音乐素养和技术水平。\n\n"
        elif overall_score >= 7.0:
            feedback += "总体评价: 您的演奏很好，展现了扎实的基本功和音乐表现力。\n\n"
        elif overall_score >= 5.5:
            feedback += "总体评价: 您的演奏基本合格，有一些亮点，但也有需要改进的地方。\n\n"
        else:
            feedback += "总体评价: 您的演奏需要更多练习，基本功和音乐性都有提升空间。\n\n"

        # 详细评分
        feedback += "详细评分:\n"
        for task, score in average_scores.items():
            feedback += f"- {task}: {score * 10:.2f}/10\n"

        feedback += "\n强项:\n"
        for task, score in average_scores.items():
            if score * 10 >= 7.0:
                feedback += f"- {task} 表现较好\n"

        feedback += "\n改进空间:\n"
        for task, score in average_scores.items():
            if score * 10 < 7.0:
                feedback += f"- {task} 需要加强\n"

        feedback += "\n改进建议:\n"
        if average_scores.get('musicality', 0) * 10 < 7.0:
            feedback += "- 加强音乐表现力，尝试更多的力度变化和音乐短语的塑造\n"
        if average_scores.get('note_accuracy', 0) * 10 < 7.0:
            feedback += "- 提高音符准确度，可以尝试慢速练习，确保每个音符都正确无误\n"
        if average_scores.get('rhythm_accuracy', 0) * 10 < 7.0:
            feedback += "- 加强节奏感，使用节拍器练习，保持稳定的节奏\n"
        if average_scores.get('tone_quality', 0) * 10 < 7.0:
            feedback += "- 改善音色，关注触键方式和踏板使用，力求音色圆润均匀\n"

        return feedback


# 使用示例
if __name__ == "__main__":
    import torch
    from model_architecture import PitchContourCNN

    # 创建模型
    model = PitchContourCNN()

    # 假设加载预训练权重（实际使用时需要加载真实权重）
    # model.load_state_dict(torch.load('models/pc_cnn_model.pth'))

    # 创建实时评估系统
    assessment_system = RealTimePianoAssessment(
        model=model,
        model_type='pc_cnn',
        task_names=['整体表现'],
        multi_task=False
    )

    # 启动交互式可视化
    assessment_system.start_interactive_visualization()

    # 启动评估
    assessment_system.start()

    # 模拟运行一段时间
    try:
        print("按Ctrl+C停止录制")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    # 停止评估
    assessment_system.stop()

    # 生成反馈
    feedback = assessment_system.generate_feedback()
    print("\n生成的反馈:")
    print(feedback)

    # 保存MIDI和音频
    assessment_system.save_midi()
    assessment_system.save_audio()

    # 显示可视化结果
    assessment_system.visualize()