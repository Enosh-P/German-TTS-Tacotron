"""
Complete Tacotron Model for German TTS
Updated to pass mel_lengths for proper masking
"""
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Tacotron(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Encoder
        self.encoder = Encoder(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            num_conv_layers=config['encoder_conv_layers'],
            conv_kernel_size=config['encoder_conv_kernel_size'],
            conv_channels=config['encoder_conv_channels'],
            lstm_hidden_size=config['encoder_lstm_hidden'],
            dropout=config['dropout']
        )
        
        # Decoder
        self.decoder = Decoder(
            n_mels=config['n_mels'],
            encoder_dim=self.encoder.output_size,
            decoder_dim=config['decoder_dim'],
            attention_dim=config['attention_dim'],
            prenet_dim=config['prenet_dim'],
            dropout=config['dropout'],
            reduction_factor=config['reduction_factor']
        )
        
        # Post-processing network (refines mel spectrogram)
        self.postnet = Postnet(
            n_mels=config['n_mels'],
            postnet_dim=config['postnet_dim'],
            num_layers=config['postnet_layers'],
            kernel_size=config['postnet_kernel_size'],
            dropout=config['dropout']
        )
    
    def forward(self, text_sequences, text_lengths, mel_targets, mel_lengths):
        """
        Training forward pass
        
        Args:
            text_sequences: (batch, max_text_len)
            text_lengths: (batch,)
            mel_targets: (batch, max_mel_len, n_mels)
            mel_lengths: (batch,) - NEW: for proper masking
        
        Returns:
            mel_outputs_before: (batch, max_mel_len, n_mels)
            mel_outputs_after: (batch, max_mel_len, n_mels)
            stop_tokens: (batch, max_mel_len//r, 1)
            alignments: (batch, max_mel_len//r, max_text_len)
        """
        # Encode text
        memory = self.encoder(text_sequences, text_lengths)
        
        # Decode to mel (now passes mel_lengths for masking)
        mel_outputs, stop_tokens, alignments = self.decoder(
            memory, mel_targets, text_lengths, mel_lengths
        )
        
        # Post-process
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet  # Residual connection
        
        return mel_outputs, mel_outputs_postnet, stop_tokens, alignments
    
    def inference(self, text_sequences, max_len=1000, stop_threshold=0.5):
        """
        Inference (generate speech)
        
        Args:
            text_sequences: (batch, max_text_len)
            max_len: Maximum mel length to generate
            stop_threshold: Stop token probability threshold (0.0-1.0)
        
        Returns:
            mel_outputs: (batch, mel_len, n_mels)
            alignments: (batch, mel_len//r, max_text_len)
        """
        # Encode
        text_lengths = torch.tensor([text_sequences.size(1)] * text_sequences.size(0))
        memory = self.encoder(text_sequences, text_lengths)
        
        # Decode (with stop token control)
        mel_outputs, alignments = self.decoder.inference(
            memory, max_len=max_len, stop_threshold=stop_threshold
        )
        
        # Post-process
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs = mel_outputs + mel_outputs_postnet
        
        return mel_outputs, alignments

class Postnet(nn.Module):
    """
    Post-processing network to refine mel spectrograms
    Stack of convolutions with residual connection
    """
    def __init__(self, n_mels=80, postnet_dim=512, num_layers=5, 
                 kernel_size=5, dropout=0.5):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_channels = n_mels if i == 0 else postnet_dim
            out_channels = n_mels if i == num_layers - 1 else postnet_dim
            
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(out_channels)
            ])
            
            if i < num_layers - 1:
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(dropout))
        
        self.convolutions = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, time, n_mels)
        Returns:
            (batch, time, n_mels)
        """
        # Conv expects (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        return x
