"""
Tacotron Decoder - Generates mel spectrograms autoregressively
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import LocationSensitiveAttention

class Prenet(nn.Module):
    """Pre-net for decoder inputs"""
    def __init__(self, in_dim, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.dropout(x, p=self.dropout, training=True)  # Always apply dropout
        x = F.relu(self.layer2(x))
        x = F.dropout(x, p=self.dropout, training=True)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        n_mels=80,
        encoder_dim=512,
        decoder_dim=1024,
        attention_dim=128,
        prenet_dim=256,
        num_lstm_layers=2,
        dropout=0.1,
        reduction_factor=2
    ):
        super().__init__()
        
        self.n_mels = n_mels
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.reduction_factor = reduction_factor
        
        # Prenet (processes previous mel frame)
        self.prenet = Prenet(n_mels * reduction_factor, prenet_dim, dropout=0.5)
        
        # Attention
        self.attention = LocationSensitiveAttention(
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
        # LSTM layers
        self.lstm1 = nn.LSTMCell(prenet_dim + encoder_dim, decoder_dim)
        self.lstm2 = nn.LSTMCell(decoder_dim, decoder_dim)
        
        # Projection to mel spectrogram
        self.mel_projection = nn.Linear(decoder_dim + encoder_dim, n_mels * reduction_factor)
        
        # Stop token prediction
        self.stop_projection = nn.Linear(decoder_dim + encoder_dim, 1)
    
    def initialize_decoder_states(self, memory, mask):
        """Initialize decoder states"""
        batch_size = memory.size(0)
        max_time = memory.size(1)
        device = memory.device
        
        # LSTM hidden states
        h1 = torch.zeros(batch_size, self.decoder_dim, device=device)
        c1 = torch.zeros(batch_size, self.decoder_dim, device=device)
        h2 = torch.zeros(batch_size, self.decoder_dim, device=device)
        c2 = torch.zeros(batch_size, self.decoder_dim, device=device)
        
        # Attention states
        attention_weights = torch.zeros(batch_size, max_time, device=device)
        attention_weights[:, 0] = 1.0  # Initialize to first position
        attention_context = torch.zeros(batch_size, self.encoder_dim, device=device)
        attention_cumulative = torch.zeros(batch_size, max_time, device=device)
        
        return (h1, c1, h2, c2), attention_context, attention_weights, attention_cumulative
    
    def decode_step(self, mel_input, memory, lstm_states, attention_context, 
                    attention_weights, attention_cumulative, mask):
        """Single decoder step"""
        h1, c1, h2, c2 = lstm_states
        
        # Prenet
        prenet_out = self.prenet(mel_input)
        
        # LSTM 1
        lstm1_input = torch.cat([prenet_out, attention_context], dim=1)
        h1, c1 = self.lstm1(lstm1_input, (h1, c1))
        h1 = F.dropout(h1, p=0.1, training=self.training)
        
        # LSTM 2
        h2, c2 = self.lstm2(h1, (h2, c2))
        h2 = F.dropout(h2, p=0.1, training=self.training)
        
        # Attention
        attention_weights_cat = torch.stack([attention_weights, attention_cumulative], dim=1)
        attention_context, attention_weights = self.attention(
            h2, memory, attention_weights_cat, mask
        )
        attention_cumulative = attention_cumulative + attention_weights
        
        # Projection
        decoder_output = torch.cat([h2, attention_context], dim=1)
        mel_output = self.mel_projection(decoder_output)
        stop_token = self.stop_projection(decoder_output)
        
        return (mel_output, stop_token, (h1, c1, h2, c2), 
                attention_context, attention_weights, attention_cumulative)
    
    def forward(self, memory, mel_targets, memory_lengths):
        """
        Args:
            memory: (batch, max_text_len, encoder_dim)
            mel_targets: (batch, max_mel_len, n_mels)
            memory_lengths: (batch,)
        
        Returns:
            mel_outputs, stop_tokens, alignments
        """
        batch_size = memory.size(0)
        max_mel_len = mel_targets.size(1)
        device = memory.device
        
        # Create mask
        mask = self._get_mask(memory_lengths, memory.size(1), device)
        
        # Initialize states
        lstm_states, attention_context, attention_weights, attention_cumulative = \
            self.initialize_decoder_states(memory, mask)
        
        # Initial input (all zeros)
        mel_input = torch.zeros(batch_size, self.n_mels * self.reduction_factor, device=device)
        
        # Output containers
        mel_outputs = []
        stop_tokens = []
        alignments = []
        
        # Decode steps
        for t in range(0, max_mel_len, self.reduction_factor):
            mel_output, stop_token, lstm_states, attention_context, attention_weights, attention_cumulative = \
                self.decode_step(mel_input, memory, lstm_states, attention_context,
                               attention_weights, attention_cumulative, mask)
            
            mel_outputs.append(mel_output)
            stop_tokens.append(stop_token)
            alignments.append(attention_weights)
            
            # Next input (teacher forcing)
            if t + self.reduction_factor < max_mel_len:
                mel_input = mel_targets[:, t:t+self.reduction_factor].reshape(batch_size, -1)
        
        mel_outputs = torch.stack(mel_outputs, dim=1)  # (batch, time/r, n_mels*r)
        stop_tokens = torch.stack(stop_tokens, dim=1)  # (batch, time/r, 1)
        alignments = torch.stack(alignments, dim=1)    # (batch, time/r, max_text_len)
        
        # Reshape mel outputs
        mel_outputs = mel_outputs.reshape(batch_size, -1, self.n_mels)
        
        return mel_outputs, stop_tokens, alignments
    
    def _get_mask(self, lengths, max_len, device):
        """Create mask for padding"""
        ids = torch.arange(0, max_len, device=device)
        mask = (ids < lengths.unsqueeze(1)).bool()
        return mask
    
    def inference(self, memory, max_len=1000):
        """Generate mel spectrogram at inference time"""
        batch_size = memory.size(0)
        device = memory.device
        
        # No mask for inference
        mask = torch.ones(batch_size, memory.size(1), device=device).bool()
        
        # Initialize
        lstm_states, attention_context, attention_weights, attention_cumulative = \
            self.initialize_decoder_states(memory, mask)
        
        mel_input = torch.zeros(batch_size, self.n_mels * self.reduction_factor, device=device)
        
        mel_outputs = []
        stop_tokens = []
        alignments = []
        
        for _ in range(max_len // self.reduction_factor):
            mel_output, stop_token, lstm_states, attention_context, attention_weights, attention_cumulative = \
                self.decode_step(mel_input, memory, lstm_states, attention_context,
                               attention_weights, attention_cumulative, mask)
            
            mel_outputs.append(mel_output)
            stop_tokens.append(stop_token)
            alignments.append(attention_weights)
            
            # Check stop condition
            if torch.sigmoid(stop_token).item() > 0.5:
                break
            
            # Use predicted output as next input
            mel_input = mel_output
        
        mel_outputs = torch.stack(mel_outputs, dim=1)
        mel_outputs = mel_outputs.reshape(batch_size, -1, self.n_mels)
        alignments = torch.stack(alignments, dim=1)
        
        return mel_outputs, alignments
