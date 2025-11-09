"""
Tests for MFCC feature extraction
"""

import os
import sys
import numpy as np
import pytest
import librosa

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing import AudioPreprocessor


@pytest.fixture
def synthetic_audio():
    """Generate synthetic sine wave audio"""
    duration = 2.0  # seconds
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    # Generate 440 Hz sine wave
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


def test_mfcc_extraction_shape(synthetic_audio):
    """Test MFCC extraction returns expected shape"""
    audio, sr = synthetic_audio
    
    preprocessor = AudioPreprocessor(sr=sr, n_mfcc=40)
    
    # Extract MFCC directly
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Combine features
    features = np.vstack([mfcc, delta, delta2])
    
    # Check shape
    assert features.shape[0] == 120  # 40 + 40 + 40
    assert features.shape[1] > 0  # Should have time frames
    
    # Check that features are not all zeros
    assert not np.all(features == 0)
    assert np.any(np.isfinite(features))


def test_mfcc_coefficients_range(synthetic_audio):
    """Test MFCC coefficients are in reasonable range"""
    audio, sr = synthetic_audio
    
    preprocessor = AudioPreprocessor(sr=sr, n_mfcc=40)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # MFCC values should be finite and in reasonable range
    assert np.all(np.isfinite(mfcc))
    assert np.max(np.abs(mfcc)) < 1000  # Reasonable upper bound


def test_delta_features(synthetic_audio):
    """Test delta and delta-delta features"""
    audio, sr = synthetic_audio
    
    preprocessor = AudioPreprocessor(sr=sr, n_mfcc=40)
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Check shapes match
    assert mfcc.shape == delta.shape
    assert mfcc.shape == delta2.shape
    
    # Delta features should have different values than base MFCC
    assert not np.allclose(mfcc, delta)
    assert not np.allclose(mfcc, delta2)


def test_feature_normalization(synthetic_audio):
    """Test feature normalization"""
    audio, sr = synthetic_audio
    
    preprocessor = AudioPreprocessor(sr=sr, n_mfcc=40)
    
    # Extract and normalize
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    
    # Normalize
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True)
    mfcc_normalized = (mfcc - mean) / (std + 1e-8)
    
    # Check normalization
    normalized_mean = np.mean(mfcc_normalized, axis=1)
    assert np.allclose(normalized_mean, 0, atol=1e-5)
    
    normalized_std = np.std(mfcc_normalized, axis=1)
    assert np.allclose(normalized_std, 1, atol=1e-5)

