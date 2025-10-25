#!/usr/bin/env python3
"""
Model verification script
Tests that the updated model loads correctly and produces expected output
"""

import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_model_architecture():
    """Test that the model creates the expected architecture"""
    print("🧠 Testing Model Architecture...")

    # Create model with new parameters
    model = DepressionDetectionModel(dropout=0.3)

    # Check model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params}")
    print(f"   Trainable parameters: {trainable_params}")
    print("   Expected parameters: ~6M")
    # Test forward pass
    batch_size = 4
    features = 120
    time_steps = 31

    x = torch.randn(batch_size, features, time_steps)
    output = model(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: {output.min().item():.3f} to {output.max().item():.3f}")

    # Check that output has correct dimensions (batch_size, 2)
    assert output.shape == (batch_size, 2), f"Expected (4, 2), got {output.shape}"
    print("   ✅ Forward pass successful")

def main():
    print("🔍 Depression Detection Model Verification")
    print("=" * 50)

    try:
        from model import DepressionDetectionModel
        from preprocessing import AudioPreprocessor

        test_model_architecture()

        print("\n" + "=" * 50)
        print("🎉 Model architecture test passed!")
        print("\n📋 Summary of improvements:")
        print("   • Reduced model complexity (fewer layers)")
        print("   • Lower dropout rates (0.4 → 0.3)")
        print("   • Enhanced data augmentation")
        print("   • Early stopping and LR scheduling")
        print("   • Cross-validation improvements")

        print("\n🚀 Next steps:")
        print("   1. Install dependencies: python install_enhanced.py")
        print("   2. Prepare data: python prepare_data.py")
        print("   3. Train model: python src/train.py --data data/training_data.csv")
        print("   4. Test API: python backend/app.py")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: python install_enhanced.py")
        return 1
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
