"""
Smoke test for inference system
Quick test to ensure model loads and can make predictions
"""

import os
import sys
import numpy as np
import soundfile as sf
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from inference import DepressionDetector


def create_test_audio():
    """Create a test audio file"""
    duration = 2.0
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    sf.write(temp_file.name, audio, sr)
    temp_file.close()
    
    return temp_file.name


def main():
    """Run smoke test"""
    print("Running smoke inference test...")
    
    # Check if model exists
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print("Skipping smoke test (model not available)")
        return 0
    
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        detector = DepressionDetector(model_path, balance_predictions=False)
        print("✓ Model loaded successfully")
        
        # Create test audio
        print("Creating test audio...")
        test_audio = create_test_audio()
        print(f"✓ Test audio created: {test_audio}")
        
        # Run prediction
        print("Running prediction...")
        prediction, confidence, probabilities = detector.predict(
            test_audio, return_probabilities=True
        )
        print(f"✓ Prediction successful")
        print(f"  Prediction: {prediction}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Probabilities: {probabilities}")
        
        # Cleanup
        if os.path.exists(test_audio):
            os.remove(test_audio)
        
        print("✓ Smoke test passed!")
        return 0
        
    except Exception as e:
        print(f"✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

