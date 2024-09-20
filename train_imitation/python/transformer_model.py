import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, num_lidar_features, num_non_lidar_features, num_actions, d_model=32, nhead=4, num_encoder_layers=3, num_patches=36):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.num_patches = num_patches  # Number of patches
        self.patch_size = num_lidar_features // self.num_patches

        # Positional Encoding for the Encoder
        self.positional_encoding = nn.Parameter(torch.zeros(self.num_patches, d_model))

        # Input Embedding for Encoder (LiDAR data)
        self.lidar_embedding = nn.Linear(self.patch_size, d_model)

        # Transformer Encoder for LiDAR data (first encoder)
        encoder_layer_1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=num_encoder_layers)

        # Input Embedding for Non-LiDAR data (values)
        self.non_lidar_embedding = nn.Linear(num_non_lidar_features, d_model)

        # Transformer Decoder (cross-attention using Q and K from LiDAR encoder and V from non-lidar data)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        # Second Encoder Layer for post-processing
        encoder_layer_2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=num_encoder_layers)

        # Linear layer to map the transformer output to actions
        self.fc_out = nn.Linear(d_model, num_actions)

    def forward(self, lidar, non_lidar):
        batch_size = lidar.size(0)
        
        # Reshape LiDAR input into patches
        lidar_patches = lidar.view(batch_size, self.num_patches, self.patch_size)

        # Linear projection of LiDAR patches and adding positional encoding
        lidar_embed = self.lidar_embedding(lidar_patches) + self.positional_encoding.unsqueeze(0)
        lidar_embed = lidar_embed.permute(1, 0, 2)  # Convert to (seq_len, batch_size, d_model)

        # Process through the transformer encoder for LiDAR data
        lidar_encoded = self.transformer_encoder_1(lidar_embed)  # Shape: (seq_len, batch_size, d_model)

        # Process non-lidar data through input embedding
        non_lidar_embed = self.non_lidar_embedding(non_lidar).unsqueeze(0)  # Shape: (1, batch_size, d_model)

        # Repeat the non-lidar embeddings along the sequence length to match the LiDAR sequence length
        non_lidar_embed = non_lidar_embed.repeat(self.num_patches, 1, 1)  # Shape: (seq_len, batch_size, d_model)

        # Cross-attention: Use LiDAR encoded data as Q and K, non-lidar as V
        non_lidar_attended, _ = self.multihead_attention(query=lidar_encoded, key=lidar_encoded, value=non_lidar_embed)

        # Process the output of the cross-attention through the second encoder layer
        encoder_output = self.transformer_encoder_2(non_lidar_attended)  # (seq_len, batch_size, d_model)

        # Aggregate over the sequence (optional, depends on your use case)
        encoder_output = encoder_output.mean(dim=0)  # Aggregate over sequence (seq_len -> batch_size, d_model)

        # Final linear layer to get the predicted actions
        actions = self.fc_out(encoder_output)
        
        return actions