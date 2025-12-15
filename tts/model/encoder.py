"""
Tacotron Encoder - Processes text sequences into hidden representations
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=512,
        num_conv_layers=3,
        conv_kernel_size=5,
        conv_channels=512,
        lstm_hidden_size=256,
        num_lstm_layers=1,
        dropout=0.5
    ):
        super().__init__()
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        conv_layers = []
        for i in range(num_conv_layers):
            in_channels = embedding_dim if i == 0 else conv_channels
            conv_layers.extend([
                nn.Conv1d(
                    in_channels,
                    conv_channels,
                    kernel_size=conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2
                ),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.convolutions = nn.Sequential(*conv_layers)
        
        # LSTM bi-directional
        self.lstm = nn.LSTM(
            conv_channels,
            lstm_hidden_size,
            num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        self.output_size = lstm_hidden_size * 2  # Bidirectional
    
    def forward(self, text_sequences, text_lengths):
        """
        Args:
            text_sequences: (batch, max_text_len) - padded text sequences
            text_lengths: (batch,) - actual lengths of sequences
        
        Returns:
            encoder_outputs: (batch, max_text_len, hidden_size)
        """
        # Embedding: (batch, max_text_len, embedding_dim)
        x = self.embedding(text_sequences)
        
        x = x.transpose(1, 2)  # Conv expects (batch, channels, time)
        x = self.convolutions(x)
        x = x.transpose(1, 2)  # Back to (batch, time, channels)
        
        # Pack padded sequence for LSTM
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        outputs, _ = self.lstm(x_packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs


