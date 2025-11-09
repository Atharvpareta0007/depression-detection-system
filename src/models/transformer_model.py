"""
Transformer model for depression detection
Lightweight Transformer encoder with positional encodings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerModel(nn.Module):
    """
    Transformer model for depression detection from speech
    
    Architecture:
    - Input projection
    - Positional encoding
    - Transformer encoder layers
    - Global pooling
    - MLP classifier head
    """
    
    def __init__(self, speech_features=120, speech_length=31,
                 d_model=128, nhead=8, num_layers=3, dim_feedforward=512,
                 dropout=0.5):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.speech_features = speech_features
        
        # Input projection
        self.input_projection = nn.Linear(speech_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=speech_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(64, 2)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, features, time_steps)
            
        Returns:
            logits: Output logits (batch_size, 2)
        """
        batch_size = x.size(0)
        
        # Project input: (batch, features, time) -> (batch, time, features)
        x = x.transpose(1, 2)
        
        # Project to d_model: (batch, time, features) -> (batch, time, d_model)
        x = self.input_projection(x)
        
        # Transpose for Transformer: (batch, time, d_model) -> (time, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Global pooling: (time, batch, d_model) -> (batch, d_model)
        # Use mean pooling
        x = x.mean(dim=0)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)
        return x

