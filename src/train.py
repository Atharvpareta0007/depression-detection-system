"""
Enhanced Training Script for Depression Detection Model
Implements early stopping, learning rate scheduling, and improved validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import json
import argparse
import sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from model import DepressionDetectionModel
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
            # Return zero features if processing fails
            features = np.zeros((120, 31), dtype=np.float32)

        # Convert to tensor and take mean across segments
        features_tensor = torch.FloatTensor(features).mean(dim=0)
        label_tensor = torch.LongTensor([label])

        return features_tensor, label_tensor


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.squeeze().to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions and labels
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Update scheduler
    scheduler.step()

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)

    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.squeeze().to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels


def train_model_with_early_stopping(train_files, train_labels, val_files, val_labels,
                                   num_epochs=100, patience=15, dropout=0.4,
                                   learning_rate=0.001, weight_decay=1e-4,
                                   output_dir='models', augment=True):
    """Train model with early stopping and learning rate scheduling"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize preprocessor
    preprocessor = AudioPreprocessor(sr=16000, n_mfcc=40, window_size=1.0, overlap=0.5)

    # Create datasets
    train_dataset = AudioDataset(train_files, train_labels, preprocessor, augment=augment)
    val_dataset = AudioDataset(val_files, val_labels, preprocessor, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Initialize model
    model = DepressionDetectionModel(dropout=dropout).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True
    )

    # Training variables
    best_val_loss = float('inf')
    best_val_accuracy = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Augmentation: {'Enabled' if augment else 'Disabled'}")

    for epoch in range(num_epochs):
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)

        # Validate
        val_loss, val_accuracy, val_precision, val_recall, val_f1, val_predictions, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'history': history
            }, os.path.join(output_dir, 'best_model.pth'))

            print(f"✓ New best model saved (Val Loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print("-" * 50)

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # Save training history
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(history, output_dir)

    return best_val_accuracy, history


def plot_training_curves(history, output_dir):
    """Plot training and validation curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_accuracy'], 'r-', label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Precision, Recall, F1
    ax3.plot(epochs, history['val_precision'], 'g-', label='Precision')
    ax3.plot(epochs, history['val_recall'], 'y-', label='Recall')
    ax3.plot(epochs, history['val_f1'], 'm-', label='F1-Score')
    ax3.set_title('Validation Metrics')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Learning rate (if available)
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def cross_validate_training(data_file, num_folds=5, **kwargs):
    """Perform k-fold cross validation"""
    print(f"Starting {num_folds}-fold cross validation...")

    # Load data (assuming CSV format with 'audio_path' and 'label' columns)
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found!")
        print("Please create a CSV file with columns: 'audio_path', 'label'")
        return

    data = pd.read_csv(data_file)
    print(f"Loaded {len(data)} samples")

    # Prepare data
    file_paths = data['audio_path'].values
    labels = data['label'].values

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []
    all_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{num_folds}")
        print(f"{'='*60}")

        # Split data
        train_files = file_paths[train_idx]
        train_labels = labels[train_idx]
        val_files = file_paths[val_idx]
        val_labels = labels[val_idx]

        print(f"Train samples: {len(train_files)}")
        print(f"Val samples: {len(val_files)}")
        print(f"Train class distribution: {np.bincount(train_labels)}")
        print(f"Val class distribution: {np.bincount(val_labels)}")

        # Train model
        best_acc, history = train_model_with_early_stopping(
            train_files, train_labels, val_files, val_labels, **kwargs
        )

        fold_results.append(best_acc)
        all_histories.append(history)

        print(f"Fold {fold + 1} completed. Best accuracy: {best_acc:.4f}")

    # Calculate statistics
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    max_acc = np.max(fold_results)

    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Best fold accuracy: {max_acc:.4f}")
    print(f"All fold accuracies: {[f'{acc:.4f}' for acc in fold_results]}")

    # Save cross-validation results
    results = {
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'best_fold_accuracy': float(max_acc),
        'fold_accuracies': [float(acc) for acc in fold_results],
        'num_folds': num_folds,
        'training_config': kwargs
    }

    with open(os.path.join(kwargs.get('output_dir', 'models'), 'cross_validation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Depression Detection Model')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--output_dir', type=str, default='models', help='Output directory')
    parser.add_argument('--no_augment', action='store_true', help='Disable data augmentation')

    args = parser.parse_args()

    # Train with cross-validation
    results = cross_validate_training(
        data_file=args.data,
        num_epochs=args.epochs,
        patience=args.patience,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        output_dir=args.output_dir,
        augment=not args.no_augment,
        num_folds=args.folds
    )

    print(f"\nTraining completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
