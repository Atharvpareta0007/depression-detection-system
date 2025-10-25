#!/usr/bin/env python3
"""
Sample script showing how to prepare data for training
"""

import pandas as pd
import os

def create_sample_data_csv():
    """Create a sample CSV file with the required format"""

    # This is just a template - you'll need to replace with your actual data
    sample_data = {
        'audio_path': [
            'data/audio/healthy_001.wav',
            'data/audio/healthy_002.wav',
            'data/audio/depressed_001.wav',
            'data/audio/depressed_002.wav',
            # Add more paths here
        ],
        'label': [
            0,  # 0 for healthy
            0,  # 0 for healthy
            1,  # 1 for depressed
            1,  # 1 for depressed
            # Add corresponding labels
        ]
    }

    df = pd.DataFrame(sample_data)
    df.to_csv('data/training_data.csv', index=False)
    print("Sample data CSV created: data/training_data.csv")
    print("\nPlease update this file with your actual audio file paths and labels:")
    print("- audio_path: Full path to .wav, .mp3, or .flac files")
    print("- label: 0 for healthy, 1 for depressed")
    print("\nExample format:")
    print(df.head())

if __name__ == "__main__":
    create_sample_data_csv()
