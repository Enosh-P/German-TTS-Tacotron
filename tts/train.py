"""
Training script for German Tacotron TTS
"""
import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np

from tts.model.tacotron import Tacotron
from tts.audio.mel import MelProcessor
from tts.dataset import get_dataloader
from tts.text.symbols import symbols

class TacotronLoss(nn.Module):
    """Combined loss for Tacotron"""
    def __init__(self):
        super().__init__()
        self.mel_loss = nn.L1Loss()
        self.stop_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, mel_pred, mel_pred_postnet, mel_target, stop_pred, stop_target):
        """
        Args:
            mel_pred: (batch, time, n_mels) - before postnet
            mel_pred_postnet: (batch, time, n_mels) - after postnet
            mel_target: (batch, time, n_mels)
            stop_pred: (batch, time//r, 1)
            stop_target: (batch, time//r, 1)
        """
        # Mel loss (before and after postnet)
        mel_loss = self.mel_loss(mel_pred, mel_target)
        mel_loss_postnet = self.mel_loss(mel_pred_postnet, mel_target)
        
        # Stop token loss
        stop_loss = self.stop_loss(stop_pred, stop_target)
        
        total_loss = mel_loss + mel_loss_postnet + stop_loss
        
        return total_loss, mel_loss, mel_loss_postnet, stop_loss

def create_stop_targets(mel_lengths, reduction_factor, max_len, device):
    """Create stop token targets"""
    batch_size = len(mel_lengths)
    stop_targets = torch.zeros(batch_size, max_len // reduction_factor, 1, device=device)
    
    for i, length in enumerate(mel_lengths):
        stop_idx = (length // reduction_factor) - 1
        if stop_idx >= 0 and stop_idx < stop_targets.size(1):
            stop_targets[i, stop_idx:, 0] = 1.0
    
    return stop_targets

def train_epoch(model, dataloader, optimizer, criterion, device, reduction_factor):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        text = batch['text'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        mel = batch['mel'].to(device)
        mel_lengths = batch['mel_lengths']
        
        # Create stop targets
        stop_targets = create_stop_targets(
            mel_lengths, reduction_factor, mel.size(1), device
        )
        
        # Forward pass
        mel_pred, mel_pred_postnet, stop_pred, alignments = model(
            text, text_lengths, mel
        )
        
        # Truncate predictions to match stop_targets length
        time_steps = stop_targets.size(1)
        mel_pred = mel_pred[:, :time_steps * reduction_factor, :]
        mel_pred_postnet = mel_pred_postnet[:, :time_steps * reduction_factor, :]
        mel_target = mel[:, :time_steps * reduction_factor, :]
        
        # Compute loss
        loss, mel_loss, mel_loss_post, stop_loss = criterion(
            mel_pred, mel_pred_postnet, mel_target, stop_pred, stop_targets
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, reduction_factor):
    """Validate model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths']
            
            stop_targets = create_stop_targets(
                mel_lengths, reduction_factor, mel.size(1), device
            )
            
            mel_pred, mel_pred_postnet, stop_pred, alignments = model(
                text, text_lengths, mel
            )
            
            time_steps = stop_targets.size(1)
            mel_pred = mel_pred[:, :time_steps * reduction_factor, :]
            mel_pred_postnet = mel_pred_postnet[:, :time_steps * reduction_factor, :]
            mel_target = mel[:, :time_steps * reduction_factor, :]
            
            loss, _, _, _ = criterion(
                mel_pred, mel_pred_postnet, mel_target, stop_pred, stop_targets
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Load config
    with open('configs/tacotron.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Add vocab size to config
    config['model']['vocab_size'] = len(symbols)
    
    # Create mel processor
    mel_processor = MelProcessor(**config['audio'])
    
    # Create dataloaders
    train_loader = get_dataloader(
        config['data']['train_metadata'],
        config['data']['mel_dir'],
        mel_processor,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True,
        reduction_factor=config['model']['reduction_factor']
    )
    
    val_loader = get_dataloader(
        config['data']['val_metadata'],
        config['data']['mel_dir'],
        mel_processor,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=False,
        reduction_factor=config['model']['reduction_factor']
    )
    
    # Create model
    model = Tacotron(config['model']).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = TacotronLoss()
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['training']['lr_step'], gamma=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config['model']['reduction_factor']
        )
        
        val_loss = validate(
            model, val_loader, criterion, device,
            config['model']['reduction_factor']
        )
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print("Saved best model!")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    main()

