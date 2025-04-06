@echo off
echo Setting up environment for QwQ-32B testing...

:: Set CUDA device if you have multiple GPUs
set CUDA_VISIBLE_DEVICES=0

:: Set PyTorch specific environment variables
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

:: Set transformers specific variables
set TRANSFORMERS_CACHE=F:/huggingface_cache
set TOKENIZERS_PARALLELISM=false

echo Running model testing script...
python test_finetune_model.py

echo Testing completed!
pause 