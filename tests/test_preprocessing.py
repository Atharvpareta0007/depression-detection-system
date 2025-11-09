"""
Tests for audio preprocessing functions
"""

import os
import sys
import numpy as np
import pytest
import soundfile as sf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import AudioPreprocessor


@pytest.fixture
def test_audio_path():
    """Path to test audio file"""
    return os.path.join(os.path.dirname(__file__), 'assets', 'test_audio.wav')


@pytest.fixture
def preprocessor():
    """Create preprocessor instance"""
    return AudioPreprocessor(sr=16000, n_mfcc=40, window_size=1.0, overlap=0.5)


def test_load_audio(test_audio_path):
    """Test loading audio file"""
    preprocessor = AudioPreprocessor()
    
    # Extract MFCC should load audio
    mfcc = preprocessor.extract_mfcc(test_audio_path, augment=False)
    
    assert mfcc is not None
    assert mfcc.shape[0] == 120  # 40 MFCC + 40 delta + 40 delta2
    assert mfcc.shape[1] > 0  # Should have time frames


def test_resample(test_audio_path):
    """Test audio resampling"""
    preprocessor = AudioPreprocessor(sr=16000)
    
    # Load and process
    mfcc = preprocessor.extract_mfcc(test_audio_path, augment=False)
    
    # Check that features were extracted (implicitly resampled to 16kHz)
    assert mfcc is not None
    assert mfcc.shape[0] == 120


def test_remove_silence(test_audio_path):
    """Test silence removal"""
    preprocessor = AudioPreprocessor()
    
    # Extract MFCC (includes silence removal)
    mfcc = preprocessor.extract_mfcc(test_audio_path, augment=False)
    
    # Should have valid features after silence removal
    assert mfcc is not None
    assert not np.all(mfcc == 0)  # Should not be all zeros


def test_segment_audio(test_audio_path, preprocessor):
    """Test audio segmentation"""
    # Process audio
    segments = preprocessor.process(test_audio_path, augment=False)
    
    assert segments is not None
    assert len(segments) > 0
    
    # Check segment shape
    assert segments[0].shape[0] == 120  # Features
    assert segments[0].shape[1] > 0  # Time frames


def test_preprocessing_pipeline(test_audio_path, preprocessor):
    """Test complete preprocessing pipeline"""
    segments = preprocessor.process(test_audio_path, augment=False)
    
    assert segments is not None
    assert len(segments) > 0
    
    # Check normalization (mean should be close to 0)
    for segment in segments:
        mean = np.mean(segment)
        std = np.std(segment)
        # After normalization, mean should be close to 0
        assert abs(mean) < 1.0  # Allow some tolerance
        assert std > 0  # Should have some variance


def test_augmentation(test_audio_path, preprocessor):
    """Test data augmentation"""
    # Process without augmentation
    segments_no_aug = preprocessor.process(test_audio_path, augment=False)
    
    # Process with augmentation
    segments_aug = preprocessor.process(test_audio_path, augment=True)
    
    assert segments_no_aug is not None
    assert segments_aug is not None
    
    # Augmented features may differ
    # Just check that both produce valid output
    assert len(segments_no_aug) > 0
    assert len(segments_aug) > 0

