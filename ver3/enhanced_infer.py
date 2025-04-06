"""
增强版钢琴表演分析模型推理脚本
这个脚本集成了结构化反馈、多维评价和NeuroPiano特征
"""

import os
import sys
import yaml
import torch
import logging
import torchaudio
import argparse
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入基础组件
from models.audio.audiomae import create_audiomae_model
from models.fusion.qformer import create_audio_qformer
from models.llm.enhanced_llm_interface import create_enhanced_audio_llm_interface
from data.preprocessing.audio_processor import AudioProcessor
from data.preprocessing.neuropiano_processor import NeuroPianoDataProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_enhanced_piano_performance_model(device="cuda"):
    """创建增强版钢琴表演分析模型"""
    # 加载配置
    with open('config/audio_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing enhanced model components...")
    
    # 初始化组件
    audiomae = create_audiomae_model()
    audiomae.to(device)
    audiomae.eval()
    
    logger.info(f"AudioMAE initialized with img_size={audiomae.img_size}")
    
    # 创建Q-former
    qformer = create_audio_qformer()
    qformer.to(device)
    qformer.eval()
    
    logger.info(f"AudioQFormer initialized")
    
    # 创建增强版LLM接口
    audio_llm = create_enhanced_audio_llm_interface()
    audio_llm.to(device)
    audio_llm.eval()
    
    # 设置audiomae和qformer为属性
    audio_llm.audiomae = audiomae
    audio_llm.qformer = qformer
    
    logger.info("Enhanced Piano Performance Model initialized")
    
    return audio_llm, config

def process_audio_file(audio_file, config, device="cuda"):
    """处理音频文件并准备模型输入"""
    logger.info(f"Processing audio file: {audio_file}")
    
    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # 如果需要重采样
    if sample_rate != config['audio_model']['sample_rate']:
        logger.info(f"Resampling from {sample_rate}Hz to {config['audio_model']['sample_rate']}Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=config['audio_model']['sample_rate']
        )
        waveform = resampler(waveform)
        sample_rate = config['audio_model']['sample_rate']
    
    # 初始化音频处理器
    audio_processor = AudioProcessor()
    
    # 创建梅尔谱图
    mel_spec = audio_processor.extract_mel_spectrogram(waveform)
    mel_spec = audio_processor.normalize_spectrogram(mel_spec)
    mel_spec = audio_processor.pad_or_truncate(mel_spec)
    
    # 如果需要，添加批次维度
    if mel_spec.dim() == 3:
        mel_spec = mel_spec.unsqueeze(0)
    
    logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
    
    return mel_spec.to(device)

def plot_mel_spectrogram(mel_spec, output_file=None):
    """绘制梅尔谱图进行可视化"""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.ylabel('Mel bands')
    plt.xlabel('Time frames')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Mel spectrogram saved to: {output_file}")
    else:
        plt.show()

def run_enhanced_inference(model, mel_spec, 
                          prompt="Please analyze this piano performance with detailed feedback on musical characteristics and performance quality.", 
                          piece_info=None, 
                          max_new_tokens=512, num_beams=5, temperature=0.7, do_sample=True, 
                          save_spectrograms=True, output_dir="results", skip_llm_generation=False):
    """运行增强版推理"""
    logger.info(f"Running enhanced inference with prompt: '{prompt}'")
    
    # 如果请求，保存可视化
    if save_spectrograms:
        os.makedirs(output_dir, exist_ok=True)
        plot_mel_spectrogram(mel_spec, os.path.join(output_dir, "input_mel_spectrogram.png"))
    
    # 完整模型推理
    with torch.no_grad():
        try:
            # 首先获取AudioMAE的嵌入
            audiomae_output = model.audiomae(mel_spec)
            logger.info(f"AudioMAE output shape: {audiomae_output.shape}")
            
            # AudioMAE输出是[B, D]，但QFormer需要[B, N, D]
            # 如果需要，添加序列维度
            if len(audiomae_output.shape) == 2:  # [B, D]
                audiomae_output = audiomae_output.unsqueeze(1)  # [B, 1, D]
                logger.info(f"AudioMAE output after adding sequence dimension: {audiomae_output.shape}")
            
            # 如果有piece_info，尝试使用NeuroPiano处理器丰富特征
            if piece_info and 'title' in piece_info:
                try:
                    logger.info(f"Enriching features with NeuroPiano data for piece: {piece_info['title']}")
                    neuropiano_processor = NeuroPianoDataProcessor()
                    neuropiano_processor.extract_all_features()
                    audiomae_output = neuropiano_processor.enrich_feature_embeddings(
                        audiomae_output, piece_info['title']
                    )
                    logger.info(f"Features enriched with NeuroPiano data")
                except Exception as e:
                    logger.warning(f"Failed to enrich features with NeuroPiano data: {e}")
            
            # 获取QFormer的嵌入
            qformer_output = model.qformer(audiomae_output)
            logger.info(f"QFormer output shape: {qformer_output.shape}")
            
            if skip_llm_generation:
                logger.info("Skipping LLM generation as requested")
                # 使用增强版结构化分析模板
                structured_analysis = model.structured_analysis_template(audiomae_output)
                return {
                    "answers": [structured_analysis],
                    "status": "success"
                }
            
            # 使用增强版LLM生成文本
            logger.info(f"Generating structured analysis with parameters: max_tokens={max_new_tokens}, beams={num_beams}, temp={temperature}")
            
            answers = model.forward(
                query_embeds=qformer_output,
                question=prompt,
                piece_info=piece_info,  # 传递乐曲信息
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample
            )
            
            logger.info("Text generation complete")
            
            return {
                "answers": answers,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

def batch_inference(model, config, audio_files, prompts=None, piece_infos=None, output_dir="results", **kwargs):
    """对多个音频文件运行推理"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认提示
    default_prompt = "Please analyze this piano performance with detailed feedback on musical characteristics and performance quality."
    
    # 设备
    device = next(model.parameters()).device
    
    results = {}
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        file_name = os.path.basename(audio_file)
        file_stem = Path(file_name).stem
        file_output_dir = os.path.join(output_dir, file_stem)
        os.makedirs(file_output_dir, exist_ok=True)
        
        logger.info(f"Processing {i+1}/{len(audio_files)}: {file_name}")
        
        # 获取此文件的提示
        prompt = prompts[i] if prompts and i < len(prompts) else default_prompt
        
        # 获取此文件的乐曲信息
        piece_info = piece_infos[i] if piece_infos and i < len(piece_infos) else None
        
        try:
            # 处理音频
            mel_spec = process_audio_file(audio_file, config, device)
            
            # 运行推理
            result = run_enhanced_inference(
                model, 
                mel_spec, 
                prompt=prompt,
                piece_info=piece_info,
                output_dir=file_output_dir,
                **kwargs
            )
            
            # 保存结果
            if result['status'] == 'success':
                answer = result['answers'][0] if result['answers'] else "No answer generated."
                
                # 确保通过使用显式utf-8编码进行正确编码
                analysis_file = os.path.join(file_output_dir, "analysis.txt")
                with open(analysis_file, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {prompt}\n\n")
                    if piece_info:
                        f.write(f"Piece Information:\n")
                        for key, value in piece_info.items():
                            f.write(f"- {key}: {value}\n")
                        f.write("\n")
                    f.write(f"Analysis:\n{answer}")
                
                # 以不同格式保存，用于验证
                with open(os.path.join(file_output_dir, "analysis.json"), "w", encoding="utf-8") as f:
                    json.dump({
                        "prompt": prompt,
                        "piece_info": piece_info,
                        "analysis": answer
                    }, f, ensure_ascii=False, indent=2)
                
                # 尝试提取结构化部分
                try:
                    sections = {}
                    current_section = None
                    section_text = []
                    
                    for line in answer.split('\n'):
                        if line.startswith('# '):
                            # 保存前一个部分
                            if current_section and section_text:
                                sections[current_section] = '\n'.join(section_text).strip()
                                section_text = []
                            
                            # 新部分开始
                            current_section = line[2:].strip()
                        elif current_section:
                            section_text.append(line)
                    
                    # 保存最后一个部分
                    if current_section and section_text:
                        sections[current_section] = '\n'.join(section_text).strip()
                    
                    # 保存结构化分析
                    if sections:
                        with open(os.path.join(file_output_dir, "structured_analysis.json"), "w", encoding="utf-8") as f:
                            json.dump(sections, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to extract structured sections: {e}")
                
                results[file_name] = {
                    "status": "success",
                    "answer": answer
                }
                
                logger.info(f"Successfully analyzed {file_name}")
            else:
                with open(os.path.join(file_output_dir, "error.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Error: {result['error']}")
                
                results[file_name] = {
                    "status": "error",
                    "error": str(result['error'])
                }
                
                logger.error(f"Failed to analyze {file_name}: {result['error']}")
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            results[file_name] = {
                "status": "error",
                "error": str(e)
            }
    
    return results

def main():
    """主函数运行推理"""
    parser = argparse.ArgumentParser(description="Enhanced Piano Performance Analysis")
    parser.add_argument("--audio", type=str, nargs="+", help="音频文件路径")
    parser.add_argument("--audio_dir", type=str, help="包含音频文件的目录")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--prompt", type=str, help="模型提示（将用于所有文件）")
    parser.add_argument("--max_tokens", type=int, default=512, help="生成的最大标记数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--num_beams", type=int, default=5, help="束搜索的束数")
    parser.add_argument("--no_spectrograms", action="store_true", help="不保存谱图")
    parser.add_argument("--with_llm", action="store_true", help="运行完整LLM生成（可能导致错误）")
    parser.add_argument("--piece_title", type=str, help="乐曲标题")
    parser.add_argument("--composer", type=str, help="作曲家")
    parser.add_argument("--style", type=str, help="音乐风格")
    parser.add_argument("--difficulty", type=str, help="难度级别")
    args = parser.parse_args()
    
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 收集音频文件
    audio_files = []
    if args.audio:
        audio_files.extend(args.audio)
    
    if args.audio_dir:
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            audio_files.extend([os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) 
                               if f.lower().endswith(ext)])
    
    if not audio_files:
        # 如果没有提供文件，使用测试音频
        test_data_dir = "test_data"
        if not os.path.exists(f"{test_data_dir}/simple_piano.wav"):
            logger.info("No audio files provided. Generating test audio...")
            try:
                from create_test_audio import main as create_audio
                create_audio()
            except Exception as e:
                logger.error(f"Failed to create test audio: {e}")
                return
        
        audio_files = [
            f"{test_data_dir}/simple_piano.wav",
            f"{test_data_dir}/complex_piano.wav"
        ]
    
    logger.info(f"Found {len(audio_files)} audio files for analysis")
    
    # 创建模型
    model, config = create_enhanced_piano_performance_model(device)
    
    # 准备乐曲信息
    piece_infos = None
    if args.piece_title or args.composer or args.style or args.difficulty:
        piece_info = {}
        if args.piece_title:
            piece_info['title'] = args.piece_title
        if args.composer:
            piece_info['composer'] = args.composer
        if args.style:
            piece_info['style'] = args.style
        if args.difficulty:
            piece_info['difficulty'] = args.difficulty
        
        # 为每个音频文件使用相同的乐曲信息
        piece_infos = [piece_info for _ in range(len(audio_files))]
    
    # 运行批量推理
    results = batch_inference(
        model, 
        config,
        audio_files, 
        prompts=[args.prompt] if args.prompt else None,
        piece_infos=piece_infos,
        output_dir=args.output_dir,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
        save_spectrograms=not args.no_spectrograms,
        skip_llm_generation=not args.with_llm  # 默认跳过LLM生成
    )
    
    # 打印摘要
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    logger.info(f"Analysis complete: {success_count}/{len(results)} files successfully analyzed")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 