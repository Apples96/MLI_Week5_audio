import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Project to full embedding dimension first
        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward (self, x):
         # x shape: (batch_size, seq_length, embed_dim)
        batch_size, seq_length, _ = x.size()

        queries = self.Wq(x)  # (batch_size, seq_length, embed_dim)
        keys = self.Wk(x)    # (batch_size, seq_length, embed_dim)
        values = self.Wv(x)  # (batch_size, seq_length, embed_dim)
        # Reshape for multi-head attention
        # Shape: (batch_size, num_heads, seq_length, head_dim)
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # For debugging, print the shapes
        sims = queries @ keys.transpose(-2, -1) / (self.head_dim ** 0.5)
        scaled_sims = F.softmax(sims, dim=-1)  # (batch_size, seq_length, seq_length)
        x = scaled_sims @ values # (batch_size, seq_length, embed_dim)

        # Reshape back to original shape
        # Shape: (batch_size, seq_length, embed_dim)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)

        x = self.output_proj(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # Layer normalization before attention and feedforward
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feedforward network
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Apply Pre-LN transformer architecture
        # First attention block with residual connection
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x
        
        # Second feedforward block with residual connection
        residual = x
        x = self.norm2(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
    



def get_sinusoid_encoding_table(max_seq_length, embed_dim):
    """
    Create sinusoidal positional embeddings.
    
    Args:
        n_position: int, maximum sequence length
        d_hid: int, embedding dimension
        
    Returns:
        torch.Tensor of shape (1, n_position, d_hid) with positional embeddings
    """
    # Create a tensor containing sequential position indices from 0 to max_seq_length-1
    # Shape: [max_seq_length, 1] - The unsqueeze creates a second dimension for broadcasting
    position = torch.arange(max_seq_length, dtype=torch.float).unsqueeze(1)

    # Create a tensor containing frequency divisors for different dimensions
    # This creates different frequencies for each embedding dimension
    # We use every 2 dimensions to accommodate both sin and cos functions
    # The frequencies decrease exponentially from 1 to 1/10000 across dimensions
    # Shape: [embed_dim/2]
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (- math.log(10000.0) / embed_dim)
    )

    # Initialize an empty tensor for positional encodings
    pe = torch.zeros(1, max_seq_length, embed_dim)
    
    # Fill even indices (0, 2, 4, ...) with sine values
    pe[0, :, 0::2] = torch.sin(position * div_term)
    
    # Fill odd indices (1, 3, 5, ...) with cosine values
    pe[0, :, 1::2] = torch.cos(position * div_term)

    return pe # Shape: [1, max_seq_length, embed_dim]




class UrbanSoundClassifier(nn.Module):
    def __init__(self, 
                max_seq_length = 1502,      # This is the max_seq_length : 1501 frames, rounded to 1502 (over 4 seconds) so can be divided by 2 
                spectrogram_channels = 128,            
                num_classes=10,         # 10 categories
                embed_dim=384,           # Embedding dimension
                num_heads=6,            # Number of attention heads
                num_layers=6,           # Number of transformer layers
                mlp_ratio=4.0,          # Ratio for MLP hidden dim
                dropout=0.1,            # Dropout rate
            ): 
        super().__init__()
        self.max_seq_length = max_seq_length
        self.spectrogram_channels = spectrogram_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        self.conv1 = nn.Conv1d(in_channels=self.spectrogram_channels, out_channels=self.embed_dim, kernel_size=3, padding = 1) 
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1) 
        
        torch.manual_seed(42)
        self.register_buffer("positional_embedding", get_sinusoid_encoding_table(self.max_seq_length//2, self.embed_dim))
        self.input_layer_norm = nn.LayerNorm(self.embed_dim)
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        # Classification head
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, log_mel_spectrogram):
        # log_mel_spectrogram shape: (batch_size, spectrogram_channels, time_frames)
        x = F.gelu(self.conv1(log_mel_spectrogram))  # Output: (batch_size, embed_dim, time_frames) # With padding=1 on both sides, the effective input length becomes 3000 + 2 = 3002. With kernel_size=3 and stride=1, the output length is (3002 - 3)/1 + 1 = 3000
        x = F.gelu(self.conv2(x)) # Output: (batch_size, embed_dim, time_frames/2) > stride of 2 divides time dimension by 2. With padding=1 on both sides, the effective input length becomes 3000 + 2 = 3002. With kernel_size=3 and stride=2, the output length is (3002 - 3)/2 + 1 = 1500. 
        # Transpose to get (batch_size, time_frames/2, embed_dim) for the transformer
        x = x.transpose(1, 2)  # Output: (batch_size, time_frames/2, embed_dim)

        # Add positional embeddings
        x = (x + self.positional_embedding).to(x.dtype)  # Output: (batch_size, time_frames/2, embed_dim)

        # Layer norm and drop out
        x = self.input_layer_norm(x) # Output: (batch_size, time_frames/2, embed_dim)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Apply final layer normalization
        x = self.final_layer_norm(x) # Output: (batch_size, time_frames/2, num_classes)
  
        # Take the prediction from the first token position (like CLS token) or average over all token positions. Here we'll use average pooling over time dimension
        x = x.mean(dim=1)  # (batch_size, num_classes)

        # Classification head
        logits = self.classifier(x)  # (batch_size, num_classes)
        
        return logits
    
