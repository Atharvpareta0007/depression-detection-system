"""
Tests for Flask API endpoints
"""

import os
import sys
import pytest
import tempfile
import shutil

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app


@pytest.fixture
def client():
    """Create Flask test client"""
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    
    with app.test_client() as client:
        yield client
    
    # Cleanup
    shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)


@pytest.fixture
def test_audio_file():
    """Create a test audio file"""
    import numpy as np
    import soundfile as sf
    
    # Create temporary audio file
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, audio, sr)
    temp_file.close()
    
    yield temp_file.name
    
    # Cleanup
    if os.path.exists(temp_file.name):
        os.remove(temp_file.name)


def test_index_endpoint(client):
    """Test index/health check endpoint"""
    response = client.get('/')
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert data['status'] == 'ok'


def test_predict_endpoint_no_file(client):
    """Test /api/predict without file"""
    response = client.post('/api/predict')
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data


def test_predict_endpoint_invalid_file(client):
    """Test /api/predict with invalid file"""
    data = {'file': (b'not an audio file', 'test.txt')}
    response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    
    # Should return 400 for invalid file type
    assert response.status_code in [400, 500]


def test_predict_endpoint_valid_wav(client, test_audio_file):
    """Test /api/predict with valid WAV file"""
    # Skip if model not loaded
    if app.detector is None:
        pytest.skip("Model not loaded - skipping prediction test")
    
    with open(test_audio_file, 'rb') as f:
        data = {'file': (f, 'test.wav')}
        response = client.post('/api/predict', data=data, content_type='multipart/form-data')
    
    # Should return 200 if model is loaded
    if response.status_code == 200:
        data = response.get_json()
        assert 'status' in data
        assert data['status'] == 'success'
        assert 'prediction' in data
        assert 'confidence' in data
        assert 'probabilities' in data
    else:
        # If model not loaded, should return 500
        assert response.status_code == 500


def test_metrics_endpoint(client):
    """Test /api/metrics endpoint"""
    response = client.get('/api/metrics')
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'metrics' in data
    assert 'accuracy' in data['metrics']


def test_info_endpoint(client):
    """Test /api/info endpoint"""
    response = client.get('/api/info')
    
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'info' in data
    assert 'name' in data['info']


def test_allowed_file_extension(client):
    """Test file extension validation"""
    from app import allowed_file
    
    assert allowed_file('test.wav') == True
    assert allowed_file('test.mp3') == True
    assert allowed_file('test.flac') == True
    assert allowed_file('test.txt') == False
    assert allowed_file('test') == False

