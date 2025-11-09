"""
Training script for different model architectures
Supports: cnn, cnn_lstm, transformer
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import DepressionDetectionModel
from models.cnn_lstm import CNNLSTMModel
from models.transformer_model import TransformerModel
from preprocessing import AudioPreprocessor


class AudioDataset(Dataset):
    """Custom dataset for audio files"""
    
    def __init__(self, file_paths, labels, preprocessor, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = preprocessor
        self.augment = augment
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Process audio
        features = self.preprocessor.process(audio_path, augment=self.augment)
        if features is None:
            features = np.zeros((120, 31), dtype=np.float32)
        
        # Convert to tensor and take mean across segments
        features_tensor = torch.FloatTensor(features).mean(dim=0)
        label_tensor = torch.LongTensor([label])
        
        return features_tensor, label_tensor


def create_model(model_type, speech_features=120, speech_length=31, dropout=0.5):
    """Create model based on type"""
    if model_type == 'cnn':
        model = DepressionDetectionModel(
            speech_features=speech_features,
            speech_length=speech_length,
            dropout=dropout
        )
    elif model_type == 'cnn_lstm':
        model = CNNLSTMModel(
            speech_features=speech_features,
            speech_length=speech_length,
            dropout=dropout
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            speech_features=speech_features,
            speech_length=speech_length,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.squeeze().to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.squeeze().to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model_type, train_files, train_labels, val_files, val_labels,
                num_epochs=50, batch_size=32, learning_rate=0.001,
                dropout=0.5, output_dir='experiments', device=None):
    """Train model"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sr=16000, n_mfcc=40, window_size=1.0, overlap=0.5)
    
    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels, preprocessor, augment=True)
    val_dataset = AudioDataset(val_files, val_labels, preprocessor, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = create_model(model_type, dropout=dropout).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training variables
    best_val_loss = float('inf')
    best_val_accuracy = 0
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': []
    }
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'model_type': model_type
            }, os.path.join(model_dir, 'best_model.pth'))
            
            print(f"  ✓ New best model saved (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
        
        print("-" * 50)
    
    # Save training history
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {model_dir}")
    
    return best_val_accuracy, history


def main():
    parser = argparse.ArgumentParser(description='Train different model architectures')
    parser.add_argument('--model', type=str, required=True, choices=['cnn', 'cnn_lstm', 'transformer'],
                       help='Model type to train')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--output_dir', type=str, default='experiments', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    
    args = parser.parse_args()
    
    # Load data
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    df = pd.read_csv(args.data)
    
    if 'audio_path' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data file must contain 'audio_path' and 'label' columns")
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    file_paths = df['audio_path'].values
    labels = df['label'].values
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels,
        test_size=args.val_ratio,
        random_state=42,
        stratify=labels
    )
    
    # Train model
    train_model(
        model_type=args.model,
        train_files=train_files,
        train_labels=train_labels,
        val_files=val_files,
        val_labels=val_labels,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Training completed for {args.model} model!")


if __name__ == "__main__":
    main()

