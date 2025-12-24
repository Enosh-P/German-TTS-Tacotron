"""
Training script for German Tacotron TTS with improved masking and stability

KEY IMPROVEMENTS:
1. Proper masking for padded sequences (both text and mel)
2. Masked loss computation (ignores padding)
3. Stop token with proper targets
4. Gradient clipping for stability
5. Alignment monitoring
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
from dataset import get_dataloader
from tts.text.symbols import symbols

class TacotronLoss(nn.Module):
    """
    Combined loss for Tacotron with proper masking
    
    MASKING EXPLANATION:
    - Padding in sequences should not contribute to loss
    - Text padding: Already handled by attention mask
    - Mel padding: Need to mask loss computation
    - Stop token: Mask loss for frames beyond sequence length
    """
    def __init__(self):
        super().__init__()
        self.mel_loss = nn.L1Loss(reduction='none')  # Don't reduce yet (need masking)
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='none')  # Don't reduce yet
    
    def forward(self, mel_pred, mel_pred_postnet, mel_target, 
                stop_pred, stop_target, mel_lengths, reduction_factor):
        """
        Args:
            mel_pred: (batch, time, n_mels) - before postnet
            mel_pred_postnet: (batch, time, n_mels) - after postnet
            mel_target: (batch, time, n_mels)
            stop_pred: (batch, time//r, 1) - raw logits
            stop_target: (batch, time//r, 1) - binary targets
            mel_lengths: (batch,) - actual mel lengths (not padded)
            reduction_factor: int - frames per decoder step
        """
        batch_size = mel_pred.size(0)
        max_mel_len = mel_pred.size(1)
        device = mel_pred.device
        
        # Create mel mask (batch, time, 1)
        mel_mask = self._create_mel_mask(mel_lengths, max_mel_len, device)
        mel_mask = mel_mask.unsqueeze(-1)  # Add feature dim
        
        # Masked mel loss (before postnet)
        mel_loss = self.mel_loss(mel_pred, mel_target)
        mel_loss = (mel_loss * mel_mask).sum() / mel_mask.sum()
        
        # Masked mel loss (after postnet)
        mel_loss_postnet = self.mel_loss(mel_pred_postnet, mel_target)
        mel_loss_postnet = (mel_loss_postnet * mel_mask).sum() / mel_mask.sum()
        
        # Create stop token mask (batch, time//r, 1)
        stop_lengths = (mel_lengths + reduction_factor - 1) // reduction_factor
        stop_mask = self._create_mel_mask(stop_lengths, stop_pred.size(1), device)
        stop_mask = stop_mask.unsqueeze(-1)
        
        # Masked stop token loss
        stop_loss = self.stop_loss(stop_pred, stop_target)
        stop_loss = (stop_loss * stop_mask).sum() / stop_mask.sum()
        
        # Total loss
        total_loss = mel_loss + mel_loss_postnet + stop_loss
        
        return total_loss, mel_loss, mel_loss_postnet, stop_loss
    
    def _create_mel_mask(self, lengths, max_len, device):
        """
        Create mask for variable length sequences
        Returns: (batch, max_len) with 1.0 for valid positions, 0.0 for padding
        """
        batch_size = lengths.size(0)
        mask = torch.zeros(batch_size, max_len, device=device)
        
        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0
        
        return mask

def create_stop_targets(mel_lengths, reduction_factor, max_len, device):
    """
    Create stop token targets with proper length handling
    
    STOP TOKEN TARGETS:
    - All frames before end: 0.0 (don't stop)
    - Last frame of sequence: 1.0 (stop here)
    - Frames after end (padding): 1.0 (but masked out in loss)
    
    This teaches the model:
    1. Keep generating until content is complete
    2. Stop at the natural end of the utterance
    3. Don't generate unnecessary frames
    """
    batch_size = len(mel_lengths)
    num_frames = max_len // reduction_factor
    stop_targets = torch.zeros(batch_size, num_frames, 1, device=device)
    
    for i, length in enumerate(mel_lengths):
        # Frame index where sequence ends
        stop_idx = (length // reduction_factor) - 1
        
        if stop_idx >= 0 and stop_idx < num_frames:
            # Set stop=1 at end of sequence
            stop_targets[i, stop_idx, 0] = 1.0
            # Also set stop=1 for all frames after (will be masked anyway)
            if stop_idx + 1 < num_frames:
                stop_targets[i, stop_idx + 1:, 0] = 1.0
    
    return stop_targets

def compute_alignment_metrics(alignments, text_lengths, mel_lengths, reduction_factor):
    """
    Compute alignment quality metrics
    
    Good alignment should be:
    1. Diagonal (monotonic progression through text)
    2. Focused (high attention on one text position)
    3. Complete (covers all text positions)
    """
    batch_size = alignments.size(0)
    
    metrics = {
        'diagonal_score': 0.0,
        'focus_score': 0.0,
        'coverage_score': 0.0
    }
    
    for i in range(batch_size):
        align = alignments[i]
        text_len = text_lengths[i]
        mel_len = mel_lengths[i] // reduction_factor
        
        # Trim to actual lengths
        align = align[:mel_len, :text_len]
        
        # Focus score (high max attention = focused)
        focus = align.max(dim=1)[0].mean().item()
        metrics['focus_score'] += focus
        
        # Coverage score (all text positions attended)
        coverage = (align.sum(dim=0) > 0.01).float().mean().item()
        metrics['coverage_score'] += coverage
        
        # Diagonal score (monotonic progression)
        expected_positions = torch.linspace(0, text_len - 1, mel_len, device=align.device)
        actual_positions = align.argmax(dim=1).float()
        diagonal = 1.0 - torch.abs(actual_positions - expected_positions).mean().item() / text_len
        metrics['diagonal_score'] += diagonal
    
    # Average over batch
    for key in metrics:
        metrics[key] /= batch_size
    
    return metrics

def train_epoch(model, dataloader, optimizer, criterion, device, reduction_factor, epoch):
    """Train for one epoch with detailed logging"""
    model.train()
    total_loss = 0
    total_mel_loss = 0
    total_stop_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        text = batch['text'].to(device)
        text_lengths = batch['text_lengths'].to(device)
        mel = batch['mel'].to(device)
        mel_lengths = batch['mel_lengths'].to(device)
        
        # Create stop targets
        stop_targets = create_stop_targets(
            mel_lengths, reduction_factor, mel.size(1), device
        )
        
        # Forward pass
        mel_pred, mel_pred_postnet, stop_pred, alignments = model(
            text, text_lengths, mel, mel_lengths
        )
        
        # Truncate to match stop_targets length
        time_steps = stop_targets.size(1)
        mel_pred = mel_pred[:, :time_steps * reduction_factor, :]
        mel_pred_postnet = mel_pred_postnet[:, :time_steps * reduction_factor, :]
        mel_target = mel[:, :time_steps * reduction_factor, :]
        mel_lengths_truncated = torch.clamp(mel_lengths, max=time_steps * reduction_factor)
        
        # Compute loss (with masking)
        loss, mel_loss, mel_loss_post, stop_loss = criterion(
            mel_pred, mel_pred_postnet, mel_target, 
            stop_pred, stop_targets,
            mel_lengths_truncated, reduction_factor
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (improves stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_mel_loss += (mel_loss.item() + mel_loss_post.item()) / 2
        total_stop_loss += stop_loss.item()
        
        # Compute alignment metrics periodically
        if batch_idx % 50 == 0:
            with torch.no_grad():
                align_metrics = compute_alignment_metrics(
                    alignments, text_lengths, mel_lengths, reduction_factor
                )
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mel': f'{mel_loss.item():.4f}',
                'stop': f'{stop_loss.item():.4f}',
                'diag': f'{align_metrics["diagonal_score"]:.3f}',
                'focus': f'{align_metrics["focus_score"]:.3f}'
            })
        else:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mel': f'{mel_loss.item():.4f}',
                'stop': f'{stop_loss.item():.4f}'
            })
    
    num_batches = len(dataloader)
    return {
        'total_loss': total_loss / num_batches,
        'mel_loss': total_mel_loss / num_batches,
        'stop_loss': total_stop_loss / num_batches
    }

def validate(model, dataloader, criterion, device, reduction_factor):
    """Validate model with masking"""
    model.eval()
    total_loss = 0
    total_mel_loss = 0
    total_stop_loss = 0
    all_align_metrics = {'diagonal_score': 0.0, 'focus_score': 0.0, 'coverage_score': 0.0}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            text = batch['text'].to(device)
            text_lengths = batch['text_lengths'].to(device)
            mel = batch['mel'].to(device)
            mel_lengths = batch['mel_lengths'].to(device)
            
            stop_targets = create_stop_targets(
                mel_lengths, reduction_factor, mel.size(1), device
            )
            
            mel_pred, mel_pred_postnet, stop_pred, alignments = model(
                text, text_lengths, mel, mel_lengths
            )
            
            time_steps = stop_targets.size(1)
            mel_pred = mel_pred[:, :time_steps * reduction_factor, :]
            mel_pred_postnet = mel_pred_postnet[:, :time_steps * reduction_factor, :]
            mel_target = mel[:, :time_steps * reduction_factor, :]
            mel_lengths_truncated = torch.clamp(mel_lengths, max=time_steps * reduction_factor)
            
            loss, mel_loss, mel_loss_post, stop_loss = criterion(
                mel_pred, mel_pred_postnet, mel_target,
                stop_pred, stop_targets,
                mel_lengths_truncated, reduction_factor
            )
            
            total_loss += loss.item()
            total_mel_loss += (mel_loss.item() + mel_loss_post.item()) / 2
            total_stop_loss += stop_loss.item()
            
            # Compute alignment metrics
            align_metrics = compute_alignment_metrics(
                alignments, text_lengths, mel_lengths, reduction_factor
            )
            for key in all_align_metrics:
                all_align_metrics[key] += align_metrics[key]
    
    num_batches = len(dataloader)
    for key in all_align_metrics:
        all_align_metrics[key] /= num_batches
    
    return {
        'total_loss': total_loss / num_batches,
        'mel_loss': total_mel_loss / num_batches,
        'stop_loss': total_stop_loss / num_batches,
        'alignment': all_align_metrics
    }

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
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device,
            config['model']['reduction_factor'], epoch + 1
        )
        
        val_metrics = validate(
            model, val_loader, criterion, device,
            config['model']['reduction_factor']
        )
        
        scheduler.step()
        
        print(f"\nTrain - Loss: {train_metrics['total_loss']:.4f}, "
              f"Mel: {train_metrics['mel_loss']:.4f}, "
              f"Stop: {train_metrics['stop_loss']:.4f}")
        print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
              f"Mel: {val_metrics['mel_loss']:.4f}, "
              f"Stop: {val_metrics['stop_loss']:.4f}")
        print(f"Alignment - Diagonal: {val_metrics['alignment']['diagonal_score']:.3f}, "
              f"Focus: {val_metrics['alignment']['focus_score']:.3f}, "
              f"Coverage: {val_metrics['alignment']['coverage_score']:.3f}")
        
        # Save checkpoint
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['total_loss'],
                'config': config
            }
            torch.save(checkpoint, 'checkpoints/best_model.pth')
            print("✓ Saved best model!")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, f'checkpoints/model_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    main()
