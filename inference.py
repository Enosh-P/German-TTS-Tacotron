"""
Inference script for German TTS with HiFi-GAN vocoder
Upgraded for high-quality neural vocoding
"""
import os
import yaml
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

from tts.model.tacotron import Tacotron
from tts.audio.mel import MelProcessor
from tts.audio.hifigan import HiFiGANVocoder
from tts.text.text_to_sequence import text_to_sequence

class TTSInference:
    def __init__(self, checkpoint_path, config_path='configs/tacotron.yaml',
                 hifigan_checkpoint='configs/generator_v3', hifigan_config='configs/config.json', use_hifigan=True):
        """
        Initialize TTS inference with optional HiFi-GAN
        
        Args:
            checkpoint_path: Path to trained Tacotron checkpoint
            config_path: Path to Tacotron config file
            hifigan_checkpoint: Path to HiFi-GAN checkpoint (e.g., 'generator_v3')
            hifigan_config: Path to HiFi-GAN config.json (optional)
            use_hifigan: Whether to use HiFi-GAN (vs Griffin-Lim)
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create mel processor
        self.mel_processor = MelProcessor(**self.config['audio'])
        
        # Load Tacotron model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        from tts.text.symbols import symbols
        self.config['model']['vocab_size'] = len(symbols)
        
        self.model = Tacotron(self.config['model']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("✓ Tacotron model loaded")
        
        # Load vocoder
        self.use_hifigan = use_hifigan and hifigan_checkpoint is not None
        
        if self.use_hifigan:
            print("\nLoading HiFi-GAN vocoder...")
            self.vocoder = HiFiGANVocoder(
                checkpoint_path=hifigan_checkpoint,
                config_path=hifigan_config,  # Can be None
                device=self.device
            )
            print("✓ HiFi-GAN vocoder ready")
        else:
            print("\nUsing Griffin-Lim vocoder (lower quality)")
            self.vocoder = None
    
    def synthesize(self, text, output_path='output.wav', max_len=1000, 
                   stop_threshold=0.5, save_alignment=True):
        """
        Synthesize speech from text
        
        Args:
            text: Input German text
            output_path: Path to save output wav file
            max_len: Maximum mel frames to generate
            stop_threshold: Stop token threshold (0.0-1.0)
            save_alignment: Save attention alignment plot
        
        Returns:
            wav: Generated waveform
            mel: Generated mel spectrogram
            alignment: Attention alignment
        """
        print(f"\n{'='*60}")
        print(f"Synthesizing: '{text}'")
        print(f"{'='*60}")
        
        # Convert text to sequence
        text_seq = text_to_sequence(text)
        print(f"Text length: {len(text_seq)} characters")
        
        text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(self.device)
        
        # Generate mel spectrogram
        print("Generating mel spectrogram...")
        with torch.no_grad():
            mel, alignment = self.model.inference(
                text_tensor, 
                max_len=max_len,
                stop_threshold=stop_threshold
            )
        
        # Convert to numpy
        mel = mel.squeeze(0).cpu().numpy()
        alignment = alignment.squeeze(0).cpu().numpy()
        
        print(f"Generated {mel.shape[0]} mel frames ({mel.shape[0] * 256 / 22050:.2f}s)")
        
        # Convert mel to waveform
        print("Converting to waveform...")
        if self.use_hifigan:
            # HiFi-GAN vocoder
            wav = self.vocoder.mel_to_wav(mel)
        else:
            # Griffin-Lim vocoder
            wav = self.mel_processor.mel_to_wav(mel)
        
        # Save audio
        sf.write(output_path, wav, self.config['audio']['sample_rate'])
        print(f"✓ Audio saved to: {output_path}")
        print(f"  Duration: {len(wav) / self.config['audio']['sample_rate']:.2f}s")
        print(f"  Sample rate: {self.config['audio']['sample_rate']} Hz")
        
        # Save alignment plot
        if save_alignment:
            alignment_path = output_path.replace('.wav', '_alignment.png')
            self._plot_alignment(alignment, text, alignment_path)
            print(f"✓ Alignment saved to: {alignment_path}")
        
        return wav, mel, alignment
    
    def synthesize_batch(self, texts, output_dir='outputs', **kwargs):
        """
        Synthesize multiple texts
        
        Args:
            texts: List of German texts
            output_dir: Directory to save outputs
            **kwargs: Additional arguments for synthesize()
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, text in enumerate(texts):
            output_path = os.path.join(output_dir, f'output_{i+1:03d}.wav')
            
            try:
                wav, mel, alignment = self.synthesize(text, output_path, **kwargs)
                results.append({
                    'text': text,
                    'wav': wav,
                    'mel': mel,
                    'alignment': alignment,
                    'path': output_path
                })
            except Exception as e:
                print(f"✗ Failed to synthesize '{text}': {e}")
                results.append(None)
        
        print(f"\n{'='*60}")
        print(f"Batch synthesis complete: {sum(r is not None for r in results)}/{len(texts)} succeeded")
        print(f"{'='*60}")
        
        return results
    
    def _plot_alignment(self, alignment, text, save_path):
        """Plot and save attention alignment"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        im = ax.imshow(alignment.T, aspect='auto', origin='lower', interpolation='none')
        ax.set_xlabel('Decoder timesteps')
        ax.set_ylabel('Encoder timesteps')
        ax.set_title(f'Attention Alignment\n"{text[:50]}..."' if len(text) > 50 else f'Attention Alignment\n"{text}"')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def analyze_alignment(self, alignment):
        """
        Analyze alignment quality
        
        Returns metrics indicating alignment quality:
        - diagonal_score: How diagonal/monotonic (0-1, higher is better)
        - focus_score: How focused attention is (0-1, higher is better)
        - coverage_score: How much of text is covered (0-1, higher is better)
        """
        # Focus: max attention value per frame
        focus = alignment.max(axis=1).mean()
        
        # Coverage: proportion of text positions attended
        coverage = (alignment.sum(axis=0) > 0.01).mean()
        
        # Diagonal score: how close to expected diagonal progression
        mel_len, text_len = alignment.shape
        expected_positions = np.linspace(0, text_len - 1, mel_len)
        actual_positions = alignment.argmax(axis=1)
        diagonal = 1.0 - np.abs(actual_positions - expected_positions).mean() / text_len
        
        return {
            'focus_score': float(focus),
            'coverage_score': float(coverage),
            'diagonal_score': float(diagonal)
        }

def main():
    """Example usage with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='German TTS Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Tacotron checkpoint')
    parser.add_argument('--text', type=str, required=True,
                       help='Text to synthesize')
    parser.add_argument('--output', type=str, default='output.wav',
                       help='Output audio file path')
    parser.add_argument('--config', type=str, default='configs/tacotron.yaml',
                       help='Path to config file')
    parser.add_argument('--hifigan', type=str, default=None,
                       help='Path to HiFi-GAN checkpoint (e.g., generator_v3)')
    parser.add_argument('--hifigan-config', type=str, default=None,
                       help='Path to HiFi-GAN config.json (optional)')
    parser.add_argument('--stop-threshold', type=float, default=0.5,
                       help='Stop token threshold (0.0-1.0)')
    parser.add_argument('--max-len', type=int, default=1000,
                       help='Maximum mel frames to generate')
    
    args = parser.parse_args()
    
    # Initialize TTS
    tts = TTSInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        hifigan_checkpoint=args.hifigan,
        hifigan_config=args.hifigan_config,
        use_hifigan=args.hifigan is not None
    )
    
    # Synthesize
    wav, mel, alignment = tts.synthesize(
        args.text, 
        args.output,
        max_len=args.max_len,
        stop_threshold=args.stop_threshold
    )
    
    # Analyze alignment
    metrics = tts.analyze_alignment(alignment)
    print(f"\nAlignment Quality:")
    print(f"  Diagonal Score: {metrics['diagonal_score']:.3f} (monotonic progression)")
    print(f"  Focus Score:    {metrics['focus_score']:.3f} (attention focus)")
    print(f"  Coverage Score: {metrics['coverage_score']:.3f} (text coverage)")
    
    # Save outputs
    np.save(args.output.replace('.wav', '_mel.npy'), mel)
    np.save(args.output.replace('.wav', '_alignment.npy'), alignment)
    
    print(f"\n{'='*60}")
    print("Synthesis complete!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
