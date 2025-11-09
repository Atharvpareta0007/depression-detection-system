"""
Tests for inference system
"""

import os
import sys
import pytest
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import DepressionDetector
from model import DepressionDetectionModel


@pytest.fixture
def test_audio_path():
    """Path to test audio file"""
    return os.path.join(os.path.dirname(__file__), 'assets', 'test_audio.wav')


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model checkpoint for testing"""
    model_path = os.path.join(tmp_path, 'mock_model.pth')
    
    # Create a small model
    model = DepressionDetectionModel(
        speech_features=120,
        speech_length=31,
        dropout=0.5
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict()
    }, model_path)
    
    return model_path


def test_inference_predict(mock_model_path, test_audio_path):
    """Test inference.predict returns expected keys"""
    detector = DepressionDetector(mock_model_path, balance_predictions=False)
    
    # Run prediction
    prediction, confidence, probabilities = detector.predict(
        test_audio_path, return_probabilities=True
    )
    
    # Check return values
    assert prediction is not None
    assert prediction in ['Healthy', 'Depressed']
    assert 0 <= confidence <= 1
    assert isinstance(probabilities, dict)
    assert 'Healthy' in probabilities
    assert 'Depressed' in probabilities
    assert abs(sum(probabilities.values()) - 1.0) < 1e-5  # Probabilities sum to 1


def test_inference_returns_json_format(mock_model_path, test_audio_path):
    """Test inference returns data in expected JSON format"""
    detector = DepressionDetector(mock_model_path, balance_predictions=False)
    
    prediction, confidence, probabilities = detector.predict(
        test_audio_path, return_probabilities=True
    )
    
    # Check that values are JSON-serializable
    assert isinstance(prediction, str)
    assert isinstance(confidence, (float, np.floating))
    assert isinstance(probabilities, dict)
    
    # Check probabilities are floats
    for key, value in probabilities.items():
        assert isinstance(value, (float, np.floating))
        assert 0 <= value <= 1


def test_inference_without_probabilities(mock_model_path, test_audio_path):
    """Test inference without returning probabilities"""
    detector = DepressionDetector(mock_model_path, balance_predictions=False)
    
    prediction, confidence = detector.predict(
        test_audio_path, return_probabilities=False
    )
    
    assert prediction is not None
    assert prediction in ['Healthy', 'Depressed']
    assert 0 <= confidence <= 1


def test_model_loading(mock_model_path):
    """Test model loads successfully"""
    detector = DepressionDetector(mock_model_path)
    
    assert detector.model is not None
    assert detector.preprocessor is not None
    assert detector.class_names == ['Healthy', 'Depressed']


def test_inference_device_selection(mock_model_path):
    """Test inference works on CPU"""
    detector = DepressionDetector(mock_model_path)
    
    # Should default to CPU if CUDA not available
    assert detector.device.type in ['cpu', 'cuda']


def test_batch_prediction(mock_model_path, test_audio_path):
    """Test batch prediction"""
    detector = DepressionDetector(mock_model_path, balance_predictions=False)
    
    # Create batch of same file
    audio_paths = [test_audio_path, test_audio_path]
    
    results = detector.predict_batch(audio_paths)
    
    assert len(results) == 2
    for result in results:
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'probabilities' in result
        assert 'error' in result

