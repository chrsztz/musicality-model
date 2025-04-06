import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Define paths
MODEL_BASE_PATH = "F:/huggingface_cache/models--Qwen--QwQ-32B/snapshots/976055f8c83f394f35dbd3ab09a285a984907bd0"
FINETUNED_MODEL_PATH = "output/qwq-32b-music-finetuned"

# Setup quantization configuration for INT4
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the base model with quantization
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE_PATH,
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_PATH, trust_remote_code=True)

# Load the fine-tuned adapter
print("Loading fine-tuned adapter...")
model = PeftModel.from_pretrained(model, FINETUNED_MODEL_PATH)

# Set up generation parameters
generation_config = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "do_sample": True
}

# Define music-related test prompts
test_prompts = [
    "Explain the concept of harmony in classical music.",
    "What are the main differences between jazz and classical piano techniques?",
    "Write a short analysis of Bach's Brandenburg Concertos.",
    "Describe the structure of a typical sonata form.",
    "Explain the Circle of Fifths and its importance in music theory.",
    "What makes Chopin's piano compositions unique?",
    "Describe the evolution of piano playing techniques from the Baroque era to the Romantic period.",
    "What are the distinct characteristics of Debussy's impressionistic music style?",
    "How does counterpoint work in musical composition?",
    "Explain the concept of musical modes and their use in different musical traditions."
]

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\n" + "="*80)
    print(f"PROMPT: {prompt}")
    print("="*80)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config,
        )
    
    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated text (remove the prompt)
    response = response[len(prompt):]
    
    print("\nRESPONSE:")
    print(response)
    print("="*80 + "\n")
    
    return response

# Test the model with each prompt
print("Starting model testing...")
for prompt in test_prompts:
    generate_response(prompt)

print("Testing completed!") 