"""
Preprocessing script for CSS10 German dataset
Converts audio files to mel spectrograms and prepares metadata
"""
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
from tts.audio.mel import MelProcessor
import yaml
import argparse

def load_dataset_metadata(dataset_dir):
    """
    Load CSS10 German dataset metadata
    CSS10 format: transcript.txt with lines like "filename|text"
    """
    transcript_path = os.path.join(dataset_dir, 'transcript.txt')
    
    samples = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                filename = parts[0]
                text = parts[1]
                
                # Get full audio path
                audio_path = os.path.join(dataset_dir, filename)
                
                if os.path.exists(audio_path):
                    samples.append({
                        'filename': filename,
                        'text': text,
                        'audio_path': audio_path
                    })
    
    return samples

def process_samples(samples, mel_processor, output_mel_dir):
    """
    Convert audio files to mel spectrograms
    """
    os.makedirs(output_mel_dir, exist_ok=True)
    
    processed = []
    failed = []
    
    print("Processing audio files...")
    for sample in tqdm(samples):
        try:
            # Load audio
            wav = mel_processor.load_wav(sample['audio_path'])
            
            # Convert to mel
            mel = mel_processor.wav_to_mel(wav)
            
            filename = sample['filename'].replace('.wav', '')
            # Save mel spectrogram
            mel_path = os.path.join(output_mel_dir, f"{filename}.npy")
            np.save(mel_path, mel)
            
            processed.append({
                'filename': filename,
                'text': sample['text'],
                'mel_path': mel_path,
                'mel_length': mel.shape[0]
            })
            
        except Exception as e:
            print(f"\nFailed to process {sample['filename']}: {e}")
            failed.append(sample['filename'])
    
    print(f"\nSuccessfully processed: {len(processed)}")
    print(f"Failed: {len(failed)}")
    
    return processed, failed

def create_metadata_splits(samples, output_dir, test_split=0.1):
    """
    Split data and create metadata files
    """
    # Shuffle samples
    random.shuffle(samples)
    
    # Split
    n_test = int(len(samples) * test_split)
    val_samples = samples[:n_test]
    train_samples = samples[n_test:]
    
    print(f"\nTrain samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Save metadata
    train_path = os.path.join(output_dir, 'train_metadata.csv')
    val_path = os.path.join(output_dir, 'val_metadata.csv')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(f"{sample['filename']}|{sample['text']}\n")
    
    with open(val_path, 'w', encoding='utf-8') as f:
        for sample in val_samples:
            f.write(f"{sample['filename']}|{sample['text']}\n")
    
    print(f"\nMetadata saved:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    
    return train_samples, val_samples

def compute_statistics(samples):
    """Compute dataset statistics"""
    text_lengths = [len(s['text']) for s in samples]
    mel_lengths = [s['mel_length'] for s in samples]
    
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(samples)}")
    print(f"\nText lengths:")
    print(f"  Min: {min(text_lengths)}")
    print(f"  Max: {max(text_lengths)}")
    print(f"  Mean: {np.mean(text_lengths):.1f}")
    print(f"\nMel lengths (frames):")
    print(f"  Min: {min(mel_lengths)}")
    print(f"  Max: {max(mel_lengths)}")
    print(f"  Mean: {np.mean(mel_lengths):.1f}")
    print(f"\nApprox audio duration:")
    print(f"  Total: {sum(mel_lengths) * 256 / 22050 / 3600:.2f} hours")

def main():
    
    parser = argparse.ArgumentParser(description='Preprocess CSS10 German dataset')
    parser.add_argument('--config', type=str, default='configs/tacotron.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup paths
    dataset_dir = config['data']['raw_dir']
    processed_dir = config['data']['processed_dir']
    mel_dir = config['data']['mel_dir']
    
    print("=" * 60)
    print("German Dataset Preprocessing")
    print("=" * 60)
    print(f"\nInput directory: {dataset_dir}")
    print(f"Output directory: {processed_dir}")
    
    # Create output directories
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    
    # Load metadata
    print("\nLoading dataset metadata...")
    samples = load_dataset_metadata(dataset_dir)
    print(f"Found {len(samples)} samples")
    
    if len(samples) == 0:
        print("\nERROR: No samples found!")
        print("Please ensure German dataset is in:", dataset_dir)
        print("Expected structure:")
        print("  raw/")
        print("    ├── transcript.txt")
        print("    └── folder/")
        print("          ├── 0001.wav")
        print("          ├── 0002.wav")
        print("          └── ...")
        return
    
    # Create mel processor
    mel_processor = MelProcessor(**config['audio'])
    
    # Process audio files
    processed_samples, failed = process_samples(samples, mel_processor, mel_dir)
    
    if len(processed_samples) == 0:
        print("\nERROR: No samples were successfully processed!")
        return
    
    # Create train/val splits
    train_samples, val_samples = create_metadata_splits(
        processed_samples,
        processed_dir,
        test_split=config['data']['test_split']
    )
    
    # Compute statistics
    compute_statistics(processed_samples)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the generated metadata files")
    print("2. Start training: python tts/train.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
