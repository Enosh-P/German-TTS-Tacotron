"""
Inference script for German TTS
"""
import os
import yaml
import torch
import soundfile as sf
import numpy as np
from tts.model.tacotron import Tacotron
from tts.audio.mel import MelProcessor
from tts.text.text_to_sequence import text_to_sequence

class TTSInference:
    def __init__(self, checkpoint_path, config_path='configs/tacotron.yaml'):
        """
        Initialize TTS inference
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to config file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create mel processor
        self.mel_processor = MelProcessor(**self.config['audio'])
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Add vocab size
        from tts.text.symbols import symbols
        self.config['model']['vocab_size'] = len(symbols)
        
        self.model = Tacotron(self.config['model']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def synthesize(self, text, output_path='output.wav', max_len=1000):
        """
        Synthesize speech from text
        
        Args:
            text: Input German text
            output_path: Path to save output wav file
            max_len: Maximum mel frames to generate
        
        Returns:
            wav: Generated waveform
            mel: Generated mel spectrogram
            alignment: Attention alignment
        """
        print(f"Synthesizing: '{text}'")
        
        # Convert text to sequence
        text_seq = text_to_sequence(text)
        text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel, alignment = self.model.inference(text_tensor, max_len=max_len)
        
        # Convert to numpy
        mel = mel.squeeze(0).cpu().numpy()
        alignment = alignment.squeeze(0).cpu().numpy()
        
        # Convert mel to waveform
        wav = self.mel_processor.mel_to_wav(mel)
        
        # Save audio
        sf.write(output_path, wav, self.config['audio']['sample_rate'])
        print(f"Audio saved to: {output_path}")
        
        return wav, mel, alignment
    
    def synthesize_batch(self, texts, output_dir='outputs'):
        """
        Synthesize multiple texts
        
        Args:
            texts: List of German texts
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f'output_{i+1}.wav').replace("\\", "/")
            self.synthesize(text, output_path)

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='German TTS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path')
    parser.add_argument('--config', type=str, default='configs/tacotron.yaml',
                       help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize TTS
    tts = TTSInference(args.checkpoint, args.config)
    
    # Synthesize
    wav, mel, alignment = tts.synthesize(args.text, args.output)
    
    # Optionally save mel and alignment
    np.save(args.output.replace('.wav', '_mel.npy'), mel)
    np.save(args.output.replace('.wav', '_alignment.npy'), alignment)
    
    print("\nSynthesis complete!")
    print(f"Mel shape: {mel.shape}")
    print(f"Alignment shape: {alignment.shape}")
    print(f"Audio duration: {len(wav) / 22050:.2f} seconds")

if __name__ == '__main__':
    main()

