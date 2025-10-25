#!/usr/bin/env python3
"""
Installation script for enhanced training dependencies
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        subprocess.check_call(cmd)
        print(f"✅ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    return True

def main():
    print("🚀 Installing Enhanced Training Dependencies")
    print("=" * 50)

    # Check Python version
    print(f"🐍 Python version: {sys.version}")

    # Install core dependencies
    success = True

    success &= run_command([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip"
    ], "Upgrading pip")

    # Install PyTorch (CPU version)
    success &= run_command([
        sys.executable, "-m", "pip", "install",
        "torch==2.5.1", "torchaudio==2.5.1",
        "-f", "https://download.pytorch.org/whl/torch_stable.html"
    ], "Installing PyTorch CPU")

    # Install other dependencies
    success &= run_command([
        sys.executable, "-m", "pip", "install",
        "numpy", "pandas", "scikit-learn", "librosa", "soundfile",
        "matplotlib", "seaborn", "plotly", "tqdm", "python-dotenv"
    ], "Installing data science packages")

    # Install Flask dependencies
    success &= run_command([
        sys.executable, "-m", "pip", "install", "flask", "flask-cors", "Pillow"
    ], "Installing web framework")

    if success:
        print("\n🎉 All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Prepare your data: python prepare_data.py")
        print("2. Train the model: python src/train.py --data data/training_data.csv")
        print("3. Run the API: python backend/app.py")
        print("4. Or run Streamlit: python run_streamlit.py")
    else:
        print("\n⚠️  Some installations failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
