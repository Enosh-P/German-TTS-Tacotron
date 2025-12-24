"""
HiFi-GAN Vocoder for high-quality waveform generation
Replaces Griffin-Lim with neural vocoder for superior audio quality
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

class ResBlock(nn.Module):
    """Residual block with dilated convolutions"""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, 
                                 dilation=d, padding=self.get_padding(kernel_size, d)))
            for d in dilation
        ])
    
    def get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, x):
        for conv in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = conv(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN Generator - Converts mel spectrogram to waveform
    
    Architecture:
    - Input: Mel spectrogram (80 channels)
    - Upsampling layers with transposed convolutions
    - Residual blocks at each resolution
    - Output: Raw waveform
    """
    def __init__(
        self,
        n_mels=80,
        upsample_rates=(8, 8, 2, 2),  # Total: 256 (matches hop_length)
        upsample_kernel_sizes=(16, 16, 4, 4),
        upsample_initial_channel=512,
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))
    ):
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Input convolution
        self.conv_pre = weight_norm(nn.Conv1d(n_mels, upsample_initial_channel, 7, 1, padding=3))
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                )
            ))
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        
        # Output convolution
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        
        # Initialize weights
        self.ups.apply(self.init_weights)
        self.conv_post.apply(self.init_weights)
    
    def init_weights(self, m):
        if type(m) == nn.Conv1d or type(m) == nn.ConvTranspose1d:
            nn.init.normal_(m.weight, 0.0, 0.01)
    
    def forward(self, mel):
        """
        Args:
            mel: (batch, n_mels, time)
        Returns:
            wav: (batch, 1, time * hop_length)
        """
        x = self.conv_pre(mel)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            
            # Apply residual blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            if xs is not None:
                x = xs / self.num_kernels
        
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference"""
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

class HiFiGANVocoder:
    """
    HiFi-GAN Vocoder Wrapper
    Provides easy interface for mel-to-wav conversion
    """
    def __init__(self, checkpoint_path=None, config_path=None, device=torch.device("cuda")):
        """
        Args:
            checkpoint_path: Path to generator checkpoint (e.g., 'generator_v3')
            config_path: Path to config.json (optional, uses defaults if not provided)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config if provided
        if config_path:
            import json
            with open(config_path, 'r') as f:
                h = json.load(f)
            
            # Create model with config parameters
            self.model = HiFiGANGenerator(
                n_mels=h.get('num_mels', 80),
                upsample_rates=h['upsample_rates'],
                upsample_kernel_sizes=h['upsample_kernel_sizes'],
                upsample_initial_channel=h['upsample_initial_channel'],
                resblock_kernel_sizes=h['resblock_kernel_sizes'],
                resblock_dilation_sizes=h['resblock_dilation_sizes']
            ).to(self.device)
            print("✓ Loaded HiFi-GAN config from:", config_path)
        else:
            # Use default architecture (works for most pretrained models)
            self.model = HiFiGANGenerator().to(self.device)
            print("✓ Using default HiFi-GAN config")
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load pretrained HiFi-GAN weights
        Handles both full checkpoint dict and direct state_dict
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'generator' in checkpoint:
                # Format: {'generator': state_dict, 'steps': ..., etc}
                state_dict = checkpoint['generator']
            elif 'model' in checkpoint:
                # Format: {'model': state_dict}
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                # Format: {'state_dict': state_dict}
                state_dict = checkpoint['state_dict']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            # Direct state_dict
            state_dict = checkpoint
        
        # Load state dict
        self.model.load_state_dict(state_dict)
        self.model.remove_weight_norm()
        print(f"✓ Loaded HiFi-GAN weights from: {checkpoint_path}")
    
    @torch.no_grad()
    def mel_to_wav(self, mel):
        """
        Convert mel spectrogram to waveform
        
        Args:
            mel: (time, n_mels) numpy array or (batch, time, n_mels) tensor
        Returns:
            wav: numpy array waveform
        """
        # Convert to tensor if needed
        if not isinstance(mel, torch.Tensor):
            mel = torch.FloatTensor(mel)
        
        # Add batch dimension if needed
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        
        # Transpose to (batch, n_mels, time)
        mel = mel.transpose(1, 2).to(self.device)
        
        # Generate waveform
        wav = self.model(mel)
        
        # Convert to numpy
        wav = wav.squeeze().cpu().numpy()
        
        return wav
    
    def denormalize_mel(self, mel, ref_level_db=20, min_level_db=-100):
        """
        Denormalize mel from [-1, 1] to proper range for HiFi-GAN
        
        Args:
            mel: Normalized mel spectrogram in [-1, 1]
        Returns:
            mel: Denormalized mel spectrogram
        """
        # Denormalize from [-1, 1] to [0, 1]
        mel = (mel + 1) / 2
        
        # Scale to dB range
        mel = mel * (ref_level_db - min_level_db) + min_level_db
        
        return mel

# Discriminators for HiFi-GAN training (if you want to train from scratch)

class PeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator"""
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
    
    def forward(self, x):
        fmap = []
        
        # Reshape to 2D
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator with multiple periods"""
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(2),
            PeriodDiscriminator(3),
            PeriodDiscriminator(5),
            PeriodDiscriminator(7),
            PeriodDiscriminator(11),
        ])
    
    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs