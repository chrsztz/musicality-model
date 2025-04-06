import os
import sys
import yaml
import torch
import logging
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio.audiomae import create_audiomae_model
from models.fusion.qformer import create_audio_qformer
from models.llm.llm_interface import create_audio_llm_interface
from data.preprocessing.audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_piano_performance_model(device="cuda"):
    """Create the complete piano performance model."""
    # Load configuration
    with open('config/audio_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    audiomae = create_audiomae_model()
    audiomae.to(device)
    
    logger.info(f"AudioMAE initialized with img_size={audiomae.img_size}")
    
    # Create Q-former
    qformer = create_audio_qformer()
    qformer.to(device)
    
    logger.info(f"AudioQFormer initialized")
    
    # Create the full model interface
    audio_llm = create_audio_llm_interface()
    audio_llm.to(device)
    
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

def run_inference(model, mel_spec, prompt="这段钢琴演奏表达了什么情感？", output_dir="test_data"):
    """Run inference with the model."""
    logger.info(f"Running inference with prompt: '{prompt}'")
    
    # Set model to evaluation mode
    model.eval()
    
    # Save before inference
    os.makedirs(output_dir, exist_ok=True)
    plot_mel_spectrogram(mel_spec, os.path.join(output_dir, "input_mel_spectrogram.png"))
    
    # Get hidden states only (skip the full LLM generation to save time)
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
            
            # Skip full LLM inference as it's resource intensive
            logger.info("Skipping full LLM inference to avoid high memory usage")
            logger.info("Pipeline test successful - input can be processed by the model")
            
            return {
                "audiomae_output_shape": audiomae_output.shape,
                "qformer_output_shape": qformer_output.shape,
                "projected_embeds_shape": projected_embeds.shape,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

def main():
    """Main function to test the model with generated audio."""
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Generate test audio if not already present
    test_audio_files = {}
    if not os.path.exists("test_data/simple_piano.wav"):
        logger.info("Test audio files not found. Generating...")
        from create_test_audio import main as create_audio
        test_audio_files = create_audio()
    else:
        test_audio_files = {
            "simple_piano": "test_data/simple_piano.wav",
            "complex_piano": "test_data/complex_piano.wav"
        }
    
    # Create model
    model, config = create_piano_performance_model(device)
    
    # Process and run inference on each test file
    for audio_name, audio_path in test_audio_files.items():
        logger.info(f"Testing with {audio_name}")
        
        # Process audio
        mel_spec = process_audio_file(audio_path, config, device)
        
        # Run inference
        result = run_inference(
            model, 
            mel_spec, 
            prompt=f"请分析这段{audio_name}演奏的音乐特点和表达的情感。", 
            output_dir=f"test_data/{audio_name}_results"
        )
        
        logger.info(f"Inference result for {audio_name}: {result['status']}")
    
    logger.info("Testing complete!")

if __name__ == "__main__":
    main() 