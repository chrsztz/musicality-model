@echo off
echo Setting up environment for QwQ-32B fine-tuning...

:: Set CUDA device if you have multiple GPUs
set CUDA_VISIBLE_DEVICES=0

:: Set PyTorch specific environment variables
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

:: Set transformers specific variables
set TRANSFORMERS_CACHE=F:/huggingface_cache
set TOKENIZERS_PARALLELISM=false

:: Optional: Enable memory tracking and analysis
set CUDA_LAUNCH_BLOCKING=1

:: Optional: Set bitsandbytes specific variables
set BNB_CUDA_VERSION=126

set HF_ENDPOINT=https://hf-mirror.com

echo Running fine-tuning script...
python music_model_finetune.py

echo Fine-tuning completed!
pause 