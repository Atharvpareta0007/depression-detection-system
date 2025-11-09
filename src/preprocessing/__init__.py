"""
Preprocessing utilities including language detection
"""

# Import AudioPreprocessor from preprocessing.py file
import sys
import os
import importlib.util

# Get the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preprocessing_file = os.path.join(parent_dir, 'preprocessing.py')

# Load AudioPreprocessor from preprocessing.py file
if os.path.exists(preprocessing_file):
    spec = importlib.util.spec_from_file_location("preprocessing_module", preprocessing_file)
    preprocessing_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preprocessing_module)
    AudioPreprocessor = preprocessing_module.AudioPreprocessor
else:
    raise ImportError("preprocessing.py file not found")

__all__ = ['AudioPreprocessor']

