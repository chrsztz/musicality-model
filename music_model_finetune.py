import os
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

# Define paths and configuration
MODEL_PATH = "F:/huggingface_cache/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
OUTPUT_DIR = "output/qwq-32b-music-finetuned"

# Setup quantization configuration for INT4
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the model with quantization
print("Loading model from:", MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Make sure padding token is set1
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare model for k-bit training
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Alpha scaling
    target_modules=[
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA config to model
model = get_peft_model(model, lora_config)
print("Trainable parameters:", model.print_trainable_parameters())

# Load MusicPile dataset
print("Loading MusicPile dataset...")
dataset = load_dataset("m-a-p/MusicPile")
train_dataset = dataset["train"]

# Function to format dataset examples
def format_dataset(example):
    # Simply use the text from the dataset
    return {
        "text": example["text"]
    }

# Apply formatting to dataset
processed_dataset = train_dataset.map(
    format_dataset,
    remove_columns=["id", "src"],  # Remove columns we don't need
    num_proc=1  # Use multiple processes for faster processing
)

# Create smaller subset for testing if needed
# processed_dataset = processed_dataset.select(range(1000))  # Uncomment to use a smaller subset

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,      # Adjust based on VRAM
    gradient_accumulation_steps=16,     # Increase for larger effective batch size
    learning_rate=2e-4,
    warmup_ratio=0.03,
    max_steps=500,                      # Adjust based on dataset size
    logging_steps=10,
    save_steps=100,
    fp16=False,                         # Using BF16 instead
    bf16=True,                          # Use BF16 for better training stability
    optim="paged_adamw_8bit",           # Memory-efficient optimizer
    seed=42,
    group_by_length=True,               # Group samples by length for efficiency
)

# Initialize SFT trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset
)

# Start training
print("Starting training...")
trainer.train()

# Save the final model
print("Saving model...")
trainer.save_model(OUTPUT_DIR)

print("Training complete!") 