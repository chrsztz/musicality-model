import os
from datasets import load_dataset, Dataset
import random
import json

# Set random seed for reproducibility
random.seed(42)

# Output directory
OUTPUT_DIR = "data/musicpile_subset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading MusicPile dataset...")
try:
    # Load the full dataset
    dataset = load_dataset("m-a-p/MusicPile")
    train_dataset = dataset["train"]
    
    # Get dataset info
    print(f"Full dataset size: {len(train_dataset)} samples")
    
    # Create a smaller subset for testing
    # We'll select 1000 random samples for testing purposes
    subset_size = 1000
    subset_indices = random.sample(range(len(train_dataset)), subset_size)
    subset = train_dataset.select(subset_indices)
    
    # Save the subset
    subset.save_to_disk(OUTPUT_DIR)
    
    # Create a sample file to inspect the data
    samples = subset.select(range(5))
    with open(os.path.join(OUTPUT_DIR, "samples.json"), "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            sample_dict = {
                "id": sample["id"],
                "text": sample["text"][:1000] + "..." if len(sample["text"]) > 1000 else sample["text"],
                "src": sample["src"],
                "text_length": len(sample["text"])
            }
            f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
    
    # Analyze subset characteristics
    text_lengths = [len(sample["text"]) for sample in subset]
    avg_length = sum(text_lengths) / len(text_lengths)
    max_length = max(text_lengths)
    min_length = min(text_lengths)
    
    print(f"Subset created with {len(subset)} samples")
    print(f"Average text length: {avg_length:.2f} characters")
    print(f"Max text length: {max_length} characters")
    print(f"Min text length: {min_length} characters")
    
    # Create a metadata file
    metadata = {
        "original_dataset": "m-a-p/MusicPile",
        "original_size": len(train_dataset),
        "subset_size": len(subset),
        "statistics": {
            "avg_length": avg_length,
            "max_length": max_length,
            "min_length": min_length
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Subset saved to {OUTPUT_DIR}")
    print(f"Sample data saved to {os.path.join(OUTPUT_DIR, 'samples.json')}")
    
except Exception as e:
    print(f"Error occurred: {e}") 