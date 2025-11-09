"""
CNN-LSTM model for depression detection
Combines CNN front-end with BiLSTM temporal pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for depression detection from speech
    
    Architecture:
    - CNN front-end for feature extraction
    - BiLSTM for temporal modeling
    - MLP classifier head
    """
    
    def __init__(self, speech_features=120, speech_length=31, 
                 lstm_hidden=128, lstm_layers=2, dropout=0.5):
        super(CNNLSTMModel, self).__init__()
        
        # CNN front-end
        self.conv1 = nn.Conv1d(speech_features, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Classifier head
        lstm_output_size = lstm_hidden * 2  # Bidirectional
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
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
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, features, time_steps)
            
        Returns:
            logits: Output logits (batch_size, 2)
        """
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Transpose for LSTM: (batch, features, time) -> (batch, time, features)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state (bidirectional, so concatenate forward and backward)
        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        batch_size = h_n.size(1)
        h_n = h_n.view(self.lstm.num_layers, 2, batch_size, self.lstm.hidden_size)
        h_forward = h_n[-1, 0]  # Last layer, forward
        h_backward = h_n[-1, 1]  # Last layer, backward
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        # Alternative: use mean pooling over time
        # h_combined = lstm_out.mean(dim=1)
        
        # Classification
        x = self.classifier(h_combined)
        return x

