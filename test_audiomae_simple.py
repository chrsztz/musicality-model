import os
import sys
import torch
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.audio.audiomae import create_audiomae_model

def main():
    # Create AudioMAE model
    print('Creating AudioMAE model...')
    audiomae = create_audiomae_model('config/audio_config.yaml')
    
    # Print model parameters
    print(f'AudioMAE img_size: {audiomae.img_size}')
    print(f'AudioMAE patch_size: {audiomae.patch_size}')
    
    # Create test input with 1024 time steps
    print('\nCreating test input with 1024 time steps...')
    n_mels = 128
    time_steps = 1024
    batch_size = 1
    
    input_tensor = torch.randn(batch_size, 1, n_mels, time_steps)
    print(f'Input shape: {input_tensor.shape}')
    
    # Resize position embeddings
    if hasattr(audiomae, 'pos_embed'):
        print(f'Original pos_embed shape: {audiomae.pos_embed.shape}')
        
        # Calculate number of patches
        h_patches = n_mels // audiomae.patch_size[0]
        w_patches = time_steps // audiomae.patch_size[1]
        n_patches = h_patches * w_patches
        print(f'Expected patches: {n_patches} (= {h_patches} * {w_patches})')
        
        # Create new position embeddings
        with torch.no_grad():
            # Save original position embeddings
            original_pos_embed = audiomae.pos_embed.clone()
            
            # Create new position embeddings
            embed_dim = audiomae.pos_embed.shape[2]
            new_pos_embed = torch.zeros(1, n_patches + 1, embed_dim)
            
            # Copy CLS token position embedding
            new_pos_embed[:, 0] = original_pos_embed[:, 0]
            
            # Initialize rest with random values
            new_pos_embed[:, 1:] = torch.randn_like(new_pos_embed[:, 1:]) * 0.02
            
            # Apply new position embeddings
            audiomae.pos_embed = torch.nn.Parameter(new_pos_embed)
            print(f'New pos_embed shape: {audiomae.pos_embed.shape}')
    
    # Forward pass
    print('\nPerforming forward pass...')
    with torch.no_grad():
        audiomae.eval()
        features, pooled = audiomae(input_tensor, return_patch_features=True)
        print(f'Patch features shape: {features.shape}')
        print(f'Pooled features shape: {pooled.shape}')
        print('Forward pass successful!')

if __name__ == '__main__':
    main() 