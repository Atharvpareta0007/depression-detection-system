#!/usr/bin/env python3
"""
Launch script for the Streamlit Depression Detection App
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Depression Detection Streamlit App")
    print("=" * 50)
    print("📊 Maximum Accuracy Model: 75%")
    print("⚖️ Balanced Predictions: No 90% bias")
    print("🎯 Enhanced Architecture: CNN + BatchNorm")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully")
    
    # Check if model exists
    model_path = "./models/best_model.pth"
    if os.path.exists(model_path):
        print(f"✅ Model found: {model_path}")
    else:
        print(f"⚠️  Model not found: {model_path}")
        print("   Make sure you're in the project root directory")
    
    print("\n🌐 Starting Streamlit server...")
    print("📱 The app will open in your browser automatically")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    main()
