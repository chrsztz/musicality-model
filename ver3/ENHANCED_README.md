# 增强版钢琴表演分析模型

这个项目是原有钢琴表演分析模型的增强版，提供了更丰富的分析功能、更全面的评估维度以及更有结构的反馈形式。

## 新增特性

### 1. 三部分结构化反馈

分析结果按照以下三部分结构组织：

- **FEEDBACK**：详细的音乐分析，包括音乐特点、技巧和整体质量
- **SUGGESTIONS**：为钢琴家提供具体、可行的建议，以改进其演奏
- **APPRECIATION**：强调演奏的优点和值得称赞的方面

### 2. 多维度评估系统

新增多个评估维度，分为客观和描述性两大类：

#### 客观评估维度
- 音高准确度 (Pitch Accuracy)
- 速度控制 (Tempo Control)
- 节奏精确度 (Rhythm Precision)
- 演奏技巧 (Articulation)
- 踏板使用 (Pedaling)
- 音色质量 (Timbre Quality)
- 力度控制 (Dynamic Control)
- 音乐平衡 (Balance)
- 完整性 (Integrity)

#### 描述性评估维度
- 对作品的理解 (Performance Understanding)
- 技术难度 (Technical Difficulty)
- 使用的技术 (Employed Techniques)
- 作曲背景 (Compositional Background)
- 情感表达 (Emotional Expression)
- 风格真实性 (Stylistic Authenticity)

### 3. NeuroPiano数据集特征集成

利用NeuroPiano数据集，提取和集成更多特征：
- 好/坏评价 (Good/Bad Evaluations)
- 问题类型 (Question Types)
- 评分 (Scores)
- 技术分析 (Technical Analysis)
- 音乐描述 (Musical Descriptions)

### 4. 乐曲信息整合

支持输入乐曲信息，生成更有针对性的分析：
- 作品标题
- 作曲家
- 音乐风格
- 难度级别

## 安装依赖

确保安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

额外依赖:

```bash
pip install nltk pandas datasets
```

## 使用方法

### 基本使用

```bash
python enhanced_infer.py
```

这将使用默认设置分析测试音频文件。

### 分析指定音频

```bash
python enhanced_infer.py --audio path/to/your/audio.wav
```

### 分析目录中的所有音频

```bash
python enhanced_infer.py --audio_dir path/to/audio/directory
```

### 添加乐曲信息

```bash
python enhanced_infer.py --audio path/to/your/audio.wav --piece_title "Moonlight Sonata" --composer "Beethoven" --style "Classical" --difficulty "Intermediate"
```

### 自定义提示和生成参数

```bash
python enhanced_infer.py --audio path/to/your/audio.wav --prompt "Analyze this piano performance focusing on technique and emotion" --max_tokens 1024 --temperature 0.8
```

### 完整参数列表

- `--audio`：指定要分析的音频文件（可以是多个）
- `--audio_dir`：指定包含要分析的音频文件的目录
- `--output_dir`：指定保存分析结果的目录（默认为"results"）
- `--prompt`：自定义提示（将用于所有文件）
- `--max_tokens`：生成的最大标记数（默认为512）
- `--temperature`：采样温度（默认为0.7）
- `--num_beams`：束搜索的束数（默认为5）
- `--no_spectrograms`：不保存梅尔谱图
- `--with_llm`：运行完整LLM生成（默认使用模板生成）
- `--piece_title`：乐曲标题
- `--composer`：作曲家
- `--style`：音乐风格
- `--difficulty`：难度级别

## 输出格式

分析结果将保存到输出目录中，每个音频文件会创建一个子目录，其中包含：

- `analysis.txt`：完整文本分析
- `analysis.json`：JSON格式的分析
- `structured_analysis.json`：按三部分结构拆分的分析
- `input_mel_spectrogram.png`：梅尔谱图可视化

## 示例分析结果

```
# FEEDBACK

This piano performance demonstrates a melancholic emotional quality with expressive playing technique in a romantic style. The recording reveals a pianist with 8/10 dynamic control capability.

The pitch accuracy is rated 9/10, indicating a solid technical foundation. The tempo control scores 7/10, showing good rhythmic awareness.

The performance has clear articulation and a well-established sense of musical phrasing. The tonal quality is rich and resonant, with good pedal control.

# SUGGESTIONS

To enhance this performance, the pianist could:

1. Explore more subtle tempo variations to highlight the musical structure
2. Consider more dramatic dynamic contrasts to enhance emotional expression
3. Refine the balance between hands, particularly in complex textures
4. Experiment with more varied articulation to highlight different musical characters

# APPRECIATION

The performance demonstrates remarkable attention to musical detail and expressive musicality. The pianist shows a deep understanding to the music's structure and emotional content.

Particularly impressive is the ability to maintain clarity in complex passages, which creates a compelling musical narrative. The subtle dynamic shadings adds significant depth to the interpretation, engaging the listener throughout the performance.
```

## 代码结构

- `enhanced_infer.py`：增强版推理脚本
- `models/llm/enhanced_llm_interface.py`：增强版LLM接口
- `data/preprocessing/neuropiano_processor.py`：NeuroPiano数据处理器

## 开发者

此增强版本在原有钢琴表演分析模型的基础上进行了扩展，旨在提供更全面、更有针对性的音乐表演分析。

## 许可证

与原项目相同。 