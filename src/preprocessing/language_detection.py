"""
Language detection for audio files
"""

import os
import sys
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import pycld3 as cld3
    CLD3_AVAILABLE = True
except ImportError:
    CLD3_AVAILABLE = False


def detect_language_from_text(text: str) -> Dict[str, any]:
    """
    Detect language from text using langdetect or pycld3
    
    Args:
        text: Text string to analyze
        
    Returns:
        Dictionary with language and confidence
    """
    if not text or len(text.strip()) == 0:
        return {
            'lang': 'unknown',
            'confidence': 0.0,
            'method': 'none'
        }
    
    # Try langdetect first
    if LANGDETECT_AVAILABLE:
        try:
            detected = langdetect.detect_langs(text)
            if detected:
                best = detected[0]
                return {
                    'lang': best.lang,
                    'confidence': float(best.prob),
                    'method': 'langdetect'
                }
        except Exception:
            pass
    
    # Try pycld3
    if CLD3_AVAILABLE:
        try:
            result = cld3.get_language(text)
            if result and result.is_reliable:
                return {
                    'lang': result.language,
                    'confidence': float(result.probability),
                    'method': 'cld3'
                }
        except Exception:
            pass
    
    # Fallback
    return {
        'lang': 'unknown',
        'confidence': 0.0,
        'method': 'fallback'
    }


def detect_language_from_audio(audio_path: str, 
                                transcript: Optional[str] = None) -> Dict[str, any]:
    """
    Detect language from audio file
    
    Args:
        audio_path: Path to audio file
        transcript: Optional transcript text
        
    Returns:
        Dictionary with language and confidence
    """
    # If transcript provided, use it
    if transcript:
        return detect_language_from_text(transcript)
    
    # For now, assume English if no transcript
    # In production, you might use an audio-language classifier
    return {
        'lang': 'en',
        'confidence': 0.5,  # Low confidence without transcript
        'method': 'assumption',
        'note': 'No transcript provided, assuming English'
    }


def is_supported_language(lang: str, supported_languages: list = None) -> bool:
    """
    Check if language is supported
    
    Args:
        lang: Language code (e.g., 'en', 'es')
        supported_languages: List of supported language codes
        
    Returns:
        True if language is supported
    """
    if supported_languages is None:
        # Default: English only
        supported_languages = ['en']
    
    return lang.lower() in [l.lower() for l in supported_languages]

