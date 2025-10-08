"""
Depression Detection System
High-accuracy speech-based depression detection using deep learning
"""

from .model import DepressionDetectionModel
from .preprocessing import AudioPreprocessor
from .inference import DepressionDetector

__version__ = "1.0.0"
__author__ = "Depression Detection Team"

__all__ = [
    'DepressionDetectionModel',
    'AudioPreprocessor', 
    'DepressionDetector'
]
