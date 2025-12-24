"""
Dataset for German TTS Training
"""
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tts.text.text_to_sequence import text_to_sequence
from tts.audio.mel import MelProcessor

class TTSDataset(Dataset):
    def __init__(self, metadata_path, mel_dir, mel_processor):
        """
        Args:
            metadata_path: Path to metadata.csv
            mel_dir: Directory containing mel spectrograms
            mel_processor: MelProcessor instance
        """
        self.mel_dir = mel_dir
        self.mel_processor = mel_processor
        
        # Load metadata
        self.samples = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    filename, text = parts[0], parts[1]
                    self.samples.append((filename, text))
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        
        # Load mel spectrogram
        mel_path = os.path.join(self.mel_dir, f"{filename}.npy")
        mel = np.load(mel_path)
        
        # Convert text to sequence
        text_seq = text_to_sequence(text)
        
        return {
            'text': np.array(text_seq, dtype=np.int64),
            'mel': mel.astype(np.float32),
            'filename': filename
        }

class TTSCollate:
    """Collate function for batching"""
    def __init__(self, reduction_factor=2):
        self.reduction_factor = reduction_factor
    
    def __call__(self, batch):
        # Sort by text length (for packing)
        batch = sorted(batch, key=lambda x: len(x['text']), reverse=True)
        
        # Get lengths
        text_lengths = [len(item['text']) for item in batch]
        mel_lengths = [item['mel'].shape[0] for item in batch]
        
        # Round mel lengths to reduction factor
        mel_lengths = [length - length % self.reduction_factor for length in mel_lengths]
        
        max_text_len = max(text_lengths)
        max_mel_len = max(mel_lengths)
        
        # Pad sequences
        batch_size = len(batch)
        n_mels = batch[0]['mel'].shape[1]
        
        text_padded = np.zeros((batch_size, max_text_len), dtype=np.int64)
        mel_padded = np.zeros((batch_size, max_mel_len, n_mels), dtype=np.float32)
        
        for i, item in enumerate(batch):
            text = item['text']
            mel = item['mel'][:mel_lengths[i]]  # Truncate to reduction factor
            
            text_padded[i, :len(text)] = text
            mel_padded[i, :mel.shape[0], :] = mel
        
        return {
            'text': torch.from_numpy(text_padded),
            'text_lengths': torch.LongTensor(text_lengths),
            'mel': torch.from_numpy(mel_padded),
            'mel_lengths': torch.LongTensor(mel_lengths)
        }

def get_dataloader(metadata_path, mel_dir, mel_processor, batch_size=32, 
                   num_workers=4, shuffle=True, reduction_factor=2):
    """Create dataloader"""
    dataset = TTSDataset(metadata_path, mel_dir, mel_processor)
    collate_fn = TTSCollate(reduction_factor=reduction_factor)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader
