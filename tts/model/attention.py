"""
Location-Sensitive Attention Mechanism for Tacotron
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocationSensitiveAttention(nn.Module):
    def __init__(
        self,
        attention_dim=128,
        encoder_dim=512,
        decoder_dim=1024,
        location_n_filters=32,
        location_kernel_size=31
    ):
        super().__init__()
        
        self.attention_dim = attention_dim
        
        # Query projection (from decoder state)
        self.query_layer = nn.Linear(decoder_dim, attention_dim, bias=False)
        
        # Key projection (from encoder outputs)
        self.memory_layer = nn.Linear(encoder_dim, attention_dim, bias=False)
        
        # Location features (from previous attention weights)
        self.location_conv = nn.Conv1d(
            2,  # Current and cumulative attention
            location_n_filters,
            kernel_size=location_kernel_size,
            padding=(location_kernel_size - 1) // 2,
            bias=False
        )
        self.location_layer = nn.Linear(location_n_filters, attention_dim, bias=False)
        
        # Energy projection
        self.energy_layer = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(self, query, memory, attention_weights_cat, mask=None):
        """
        Args:
            query: (batch, decoder_dim) - current decoder state
            memory: (batch, max_text_len, encoder_dim) - encoder outputs
            attention_weights_cat: (batch, 2, max_text_len) - [prev_attn, cumulative_attn]
            mask: (batch, max_text_len) - mask for padding
        
        Returns:
            context: (batch, encoder_dim)
            attention_weights: (batch, max_text_len)
        """
        # Process query
        processed_query = self.query_layer(query.unsqueeze(1))  # (batch, 1, attention_dim)
        
        # Process memory
        processed_memory = self.memory_layer(memory)  # (batch, max_text_len, attention_dim)
        
        # Process location features
        location_features = self.location_conv(attention_weights_cat)  # (batch, filters, max_text_len)
        location_features = location_features.transpose(1, 2)  # (batch, max_text_len, filters)
        processed_location = self.location_layer(location_features)  # (batch, max_text_len, attention_dim)
        
        # Compute energies
        energies = self.energy_layer(torch.tanh(
            processed_query + processed_memory + processed_location
        )).squeeze(-1)  # (batch, max_text_len)
        
        # Apply mask
        if mask is not None:
            energies = energies.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(energies, dim=1)  # (batch, max_text_len)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)  # (batch, encoder_dim)
        
        return context, attention_weights