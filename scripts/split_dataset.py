"""
Stratified train/val/test split with speaker-level separation
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def split_dataset(data_file, output_dir='data', 
                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  stratify=True, speaker_column=None, random_state=42):
    """
    Perform stratified train/val/test split
    
    Args:
        data_file: Path to CSV file with 'audio_path' and 'label' columns
        output_dir: Output directory for split files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        stratify: Whether to stratify by label
        speaker_column: Optional column name for speaker ID (for speaker-level separation)
        random_state: Random seed
    """
    # Load data
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Check required columns
    if 'audio_path' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data file must contain 'audio_path' and 'label' columns")
    
    print(f"Loaded {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Speaker-level separation if speaker column provided
    if speaker_column and speaker_column in df.columns:
        print(f"Using speaker-level separation with column: {speaker_column}")
        
        # Get unique speakers
        speakers = df[speaker_column].unique()
        print(f"Found {len(speakers)} unique speakers")
        
        # Split speakers
        speakers_train, speakers_temp = train_test_split(
            speakers, test_size=(1 - train_ratio), random_state=random_state
        )
        speakers_val, speakers_test = train_test_split(
            speakers_temp, 
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=random_state
        )
        
        # Split data by speakers
        train_df = df[df[speaker_column].isin(speakers_train)]
        val_df = df[df[speaker_column].isin(speakers_val)]
        test_df = df[df[speaker_column].isin(speakers_test)]
        
        print(f"Train: {len(train_df)} samples ({len(speakers_train)} speakers)")
        print(f"Val: {len(val_df)} samples ({len(speakers_val)} speakers)")
        print(f"Test: {len(test_df)} samples ({len(speakers_test)} speakers)")
        
    else:
        # Standard stratified split
        print("Using standard stratified split")
        
        # First split: train vs temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df['label'] if stratify else None,
            random_state=random_state
        )
        
        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_df['label'] if stratify else None,
            random_state=random_state
        )
        
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
    
    # Check class distribution
    print("\nClass distribution:")
    print(f"Train: {train_df['label'].value_counts().to_dict()}")
    print(f"Val: {val_df['label'].value_counts().to_dict()}")
    print(f"Test: {test_df['label'].value_counts().to_dict()}")
    
    # Save splits
    train_file = os.path.join(output_dir, 'train.csv')
    val_file = os.path.join(output_dir, 'val.csv')
    test_file = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"\nSplits saved to:")
    print(f"  Train: {train_file}")
    print(f"  Val: {val_file}")
    print(f"  Test: {test_file}")
    
    # Save split info
    split_info = {
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_ratio': len(train_df) / len(df),
        'val_ratio': len(val_df) / len(df),
        'test_ratio': len(test_df) / len(df),
        'class_distribution': {
            'train': train_df['label'].value_counts().to_dict(),
            'val': val_df['label'].value_counts().to_dict(),
            'test': test_df['label'].value_counts().to_dict()
        },
        'random_state': random_state
    }
    
    info_file = os.path.join(output_dir, 'split_info.json')
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"  Split info: {info_file}")
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test')
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--speaker_column', type=str, default=None, help='Speaker ID column name')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--no_stratify', action='store_true', help='Disable stratification')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Split dataset
    split_dataset(
        data_file=args.data,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=not args.no_stratify,
        speaker_column=args.speaker_column,
        random_state=args.random_state
    )
    
    print("\n✓ Dataset split completed!")


if __name__ == "__main__":
    main()

