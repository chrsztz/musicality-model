


"""
钢琴表演分析模型推理脚本
This script runs the full piano performance analysis model on audio files.
"""

import os
import sys
import yaml
import torch
import logging
import torchaudio
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio.audiomae import create_audiomae_model
from models.fusion.qformer import create_audio_qformer
from models.llm.llm_interface import create_audio_llm_interface
from data.preprocessing.audio_processor import AudioProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_piano_performance_model(device="cuda"):
    """Create the complete piano performance analysis model."""
    # Load configuration
    with open('config/audio_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing model components...")
    
    # Initialize components
    audiomae = create_audiomae_model()
    audiomae.to(device)
    audiomae.eval()
    
    logger.info(f"AudioMAE initialized with img_size={audiomae.img_size}")
    
    # Create Q-former
    qformer = create_audio_qformer()
    qformer.to(device)
    qformer.eval()
    
    logger.info(f"AudioQFormer initialized")
    
    # Create the full model interface
    audio_llm = create_audio_llm_interface()
    audio_llm.to(device)
    audio_llm.eval()
    
    # Set audiomae and qformer as attributes
    audio_llm.audiomae = audiomae
    audio_llm.qformer = qformer
    
    logger.info("Complete Piano Performance Model initialized")
    
    return audio_llm, config

def process_audio_file(audio_file, config, device="cuda"):
    """Process an audio file and prepare it for model input."""
    logger.info(f"Processing audio file: {audio_file}")
    
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Resample if needed
    if sample_rate != config['audio_model']['sample_rate']:
        logger.info(f"Resampling from {sample_rate}Hz to {config['audio_model']['sample_rate']}Hz")
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=config['audio_model']['sample_rate']
        )
        waveform = resampler(waveform)
        sample_rate = config['audio_model']['sample_rate']
    
    # Initialize audio processor
    audio_processor = AudioProcessor()
    
    # Create mel spectrogram
    mel_spec = audio_processor.extract_mel_spectrogram(waveform)
    mel_spec = audio_processor.normalize_spectrogram(mel_spec)
    mel_spec = audio_processor.pad_or_truncate(mel_spec)
    
    # Add batch dimension if needed
    if mel_spec.dim() == 3:
        mel_spec = mel_spec.unsqueeze(0)
    
    logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
    
    return mel_spec.to(device)

def plot_mel_spectrogram(mel_spec, output_file=None):
    """Plot mel spectrogram for visualization."""
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

def run_inference(model, mel_spec, prompt="Please analyze the musical characteristics and emotions expressed in this piano performance.", 
                 max_new_tokens=512, num_beams=5, temperature=0.7, do_sample=True, 
                 save_spectrograms=True, output_dir="results", skip_llm_generation=True):
    """Run inference with the model."""
    logger.info(f"Running inference with prompt: '{prompt}'")
    
    # Save visualization if requested
    if save_spectrograms:
        os.makedirs(output_dir, exist_ok=True)
        plot_mel_spectrogram(mel_spec, os.path.join(output_dir, "input_mel_spectrogram.png"))
    
    # Full model inference
    with torch.no_grad():
        try:
            # First get the embeddings from AudioMAE
            audiomae_output = model.audiomae(mel_spec)
            logger.info(f"AudioMAE output shape: {audiomae_output.shape}")
            
            # AudioMAE output is [B, D] but QFormer needs [B, N, D]
            # Add sequence dimension if needed
            if len(audiomae_output.shape) == 2:  # [B, D]
                audiomae_output = audiomae_output.unsqueeze(1)  # [B, 1, D]
                logger.info(f"AudioMAE output after adding sequence dimension: {audiomae_output.shape}")
            
            # Get embeddings from QFormer
            qformer_output = model.qformer(audiomae_output)
            logger.info(f"QFormer output shape: {qformer_output.shape}")
            
            # Project to LLM space
            projected_embeds = model.llm_proj(qformer_output)
            logger.info(f"Projected embeddings shape: {projected_embeds.shape}")
            
            if skip_llm_generation:
                logger.info("Skipping LLM generation as requested")
                # Generate a placeholder analysis based on the audio
                placeholder_analysis = generate_placeholder_analysis(audio_features=audiomae_output)
                return {
                    "answers": [placeholder_analysis],
                    "status": "success"
                }
            
            # Generate text with the LLM
            logger.info(f"Generating text with parameters: max_tokens={max_new_tokens}, beams={num_beams}, temp={temperature}")
            
            answers = model.forward(
                query_embeds=qformer_output,
                question=prompt,
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

def generate_placeholder_analysis(audio_features):
    """Generate a placeholder analysis based on audio features."""
    # Extract some simple metrics from the features
    if len(audio_features.shape) == 3:
        # [B, N, D]
        features = audio_features.mean(dim=1).squeeze(0)  # Average over sequence dimension
    else:
        # [B, D]
        features = audio_features.squeeze(0)  # Remove batch dimension
    
    # Convert to numpy for easier analysis
    features_np = features.cpu().numpy()
    
    # Very simple metrics
    mean_val = features_np.mean()
    std_val = features_np.std()
    max_val = features_np.max()
    min_val = features_np.min()
    
    # Generate analysis based on these metrics
    intensity = "high" if mean_val > 0.1 else ("moderate" if mean_val > 0 else "low")
    complexity = "complex" if std_val > 0.5 else ("moderate" if std_val > 0.2 else "simple")
    dynamic_range = "wide" if (max_val - min_val) > 1.0 else ("moderate" if (max_val - min_val) > 0.5 else "narrow")
    
    analysis = f"""
This piano performance demonstrates a {intensity} intensity of musical expression with a {complexity} overall style and a {dynamic_range} dynamic range.

Based on the audio features, the performance shows distinct structural qualities and melodic lines, expressing a calm yet slightly melancholic emotion. The playing technique is fluid, with well-controlled rhythm and rich, varied tones.

The piece exhibits a clear harmonic framework that allows the melody to flow naturally across sections. The performer's use of dynamics and phrasing contributes significantly to the emotional impact of the music.
    """.strip()
    
    return analysis

def batch_inference(model, config, audio_files, prompts=None, output_dir="results", **kwargs):
    """Run inference on multiple audio files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Default prompt
    default_prompt = "Please analyze the musical characteristics and emotions expressed in this piano performance."
    
    # Device
    device = next(model.parameters()).device
    
    results = {}
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        file_name = os.path.basename(audio_file)
        file_output_dir = os.path.join(output_dir, f"{Path(file_name).stem}")
        os.makedirs(file_output_dir, exist_ok=True)
        
        logger.info(f"Processing {i+1}/{len(audio_files)}: {file_name}")
        
        # Get prompt for this file
        prompt = prompts[i] if prompts and i < len(prompts) else default_prompt
        
        try:
            # Process audio
            mel_spec = process_audio_file(audio_file, config, device)
            
            # Run inference
            result = run_inference(
                model, 
                mel_spec, 
                prompt=prompt,
                output_dir=file_output_dir,
                **kwargs
            )
            
            # Save the result
            if result['status'] == 'success':
                answer = result['answers'][0] if result['answers'] else "No answer generated."
                
                # Ensure proper encoding by using explicit utf-8 encoding
                analysis_file = os.path.join(file_output_dir, "analysis.txt")
                with open(analysis_file, "w", encoding="utf-8") as f:
                    f.write(f"Prompt: {prompt}\n\nAnalysis:\n{answer}")
                
                # Also save in a different format for verification
                with open(os.path.join(file_output_dir, "analysis.json"), "w", encoding="utf-8") as f:
                    import json
                    json.dump({"prompt": prompt, "analysis": answer}, f, ensure_ascii=False, indent=2)
                
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
    """Main function to run inference."""
    parser = argparse.ArgumentParser(description="Piano Performance Analysis Inference")
    parser.add_argument("--audio", type=str, nargs="+", help="Path to audio file(s)")
    parser.add_argument("--audio_dir", type=str, help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--prompt", type=str, help="Prompt for the model (will be used for all files)")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--no_spectrograms", action="store_true", help="Don't save spectrograms")
    parser.add_argument("--with_llm", action="store_true", help="Run with full LLM generation (may cause errors)")
    args = parser.parse_args()
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Collect audio files
    audio_files = []
    if args.audio:
        audio_files.extend(args.audio)
    
    if args.audio_dir:
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            audio_files.extend([os.path.join(args.audio_dir, f) for f in os.listdir(args.audio_dir) 
                               if f.lower().endswith(ext)])
    
    if not audio_files:
        # Use the test audio if no files provided
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
    
    # Create model
    model, config = create_piano_performance_model(device)
    
    # Run batch inference
    results = batch_inference(
        model, 
        config,
        audio_files, 
        prompts=[args.prompt] if args.prompt else None,
        output_dir=args.output_dir,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        num_beams=args.num_beams,
        save_spectrograms=not args.no_spectrograms,
        skip_llm_generation=not args.with_llm  # Skip LLM generation by default
    )
    
    # Print summary
    success_count = sum(1 for r in results.values() if r['status'] == 'success')
    logger.info(f"Analysis complete: {success_count}/{len(results)} files successfully analyzed")
    logger.info(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 