import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)
    def forward (self, x):
         # Input x shape: (batch_size, seq_length, embed_dim)
         batch_size = x.size(0)

         query_emb = self.Wq(x)  # (batch_size, seq_length, embed_dim)
         key_emb = self.Wk(x)    # (batch_size, seq_length, embed_dim)
         value_emb = self.Wv(x)  # (batch_size, seq_length, embed_dim)

         # For debugging, print the shapes
         sims = query_emb @ key_emb.transpose(1, 2) / (self.embed_dim ** 0.5)
         scaled_sims = F.softmax(sims, dim=-1)  # (batch_size, seq_length, seq_length)
         x = scaled_sims @ value_emb # (batch_size, seq_length, embed_dim)
         return x


class UrbanSoundClassifier(nn.Module):
    def __init__(self, 
                max_seq_length = 1501,      # This is the max_seq_length : 1501 frames (over 4 seconds) 
                spectrogram_channels = 128,            
                num_classes=10,         # 10 categories
                embed_dim=64,           # Embedding dimension
                num_heads=1,            # Number of attention heads
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
        
        self.conv1 = nn.Conv1d(in_channels=self.spectrogram_channels, 
                               out_channels=self.embed_dim, 
                               kernel_size=1)
        
        torch.manual_seed(42)
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_length, self.embed_dim)) # Made positional embeddings a proper nn.Parameter so it's saved with the model, Added batch dimension to match expected shape: (1, max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.attention = SelfAttention(self.embed_dim)
        self.ff1 = nn.Linear(self.embed_dim, self.embed_dim) 
        self.ff2 = nn.Linear(self.embed_dim, self.num_classes)
        
        self.dropout = nn.Dropout(dropout)



    def forward(self, log_mel_spectrogram):
        # log_mel_spectrogram shape: (batch_size, spectrogram_channels, time_frames)
        x = self.conv1(log_mel_spectrogram)  # Output: (batch_size, embed_dim, time_frames)
        # Transpose to get (batch_size, time_frames, embed_dim) for the transformer
        x = x.transpose(1, 2)  # Output: (batch_size, time_frames, embed_dim)

        # Add positional embeddings
        # We use the positional embeddings up to the sequence length
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]  # Output: (batch_size, time_frames, embed_dim)
        x = self.layer_norm(x) # Output: (batch_size, time_frames, embed_dim)
        x = self.dropout(x)
        
        for i in range(self.num_layers):
                residuals1 = x # Output: (batch_size, time_frames, embed_dim)
                x = self.attention(x) # Output: (batch_size, time_frames, embed_dim)
                x = x + residuals1 # Output: (batch_size, time_frames, embed_dim)
                x = self.layer_norm(x) # Output: (batch_size, time_frames, embed_dim)
                
                residuals2 = x # Output: (batch_size, time_frames, embed_dim)
                x = self.ff1(x) # Output: (batch_size, time_frames, embed_dim)
                x = x + residuals2 # Output: (batch_size, time_frames, embed_dim)
                x = self.layer_norm(x) # Output: (batch_size, time_frames, embed_dim)

        x = self.ff2(x)  # Output: (batch_size, time_frames, num_classes)
        # Take the prediction from the first token position (like CLS token) or average over all token positions. Here we'll use average pooling over time dimension
        logits = x.mean(dim=1)  # (batch_size, num_classes)
        
        return logits
