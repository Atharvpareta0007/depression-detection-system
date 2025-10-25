"""
Audio preprocessing for depression detection
Extracts MFCC features from speech audio
"""

import numpy as np
import librosa
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Preprocesses audio files for depression detection
    Extracts MFCC features with delta and delta-delta coefficients
    """

    def __init__(self, sr=16000, n_mfcc=40, n_fft=2048, hop_length=512,
                 window_size=1.0, overlap=0.5):
        """
        Args:
            sr: Target sampling rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Number of samples between successive frames
            window_size: Window size in seconds
            overlap: Overlap ratio for segmentation
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
        self.overlap = overlap
        self.window_samples = int(window_size * sr)
        self.step_samples = int(self.window_samples * (1 - overlap))

    def extract_mfcc(self, audio_path, augment=False):
        """
        Extract MFCC features from audio file

        Args:
            audio_path: Path to audio file
            augment: Whether to apply data augmentation (for training)

        Returns:
            MFCC features with delta and delta-delta (n_features x time_frames)
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)

            # Remove silence
            y, _ = librosa.effects.trim(y, top_db=20)

            # Apply augmentation for training
            if augment:
                augmenter = AudioAugmentation(sr)
                y = augmenter.augment_audio(y)

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            # Delta and delta-delta features
            delta = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            # Combine features (40 + 40 + 40 = 120 features)
            features = np.vstack([mfcc, delta, delta2])

            return features

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def segment_features(self, features):
        """
        Segment MFCC features into fixed-length windows

        Args:
            features: MFCC features (n_features x time_frames)

        Returns:
            Segmented features (num_segments x n_features x frames_per_window)
        """
        n_features, n_frames = features.shape
        frames_per_window = int(self.window_size * self.sr / self.hop_length)
        step_frames = int(frames_per_window * (1 - self.overlap))

        segments = []
        start = 0

        while start + frames_per_window <= n_frames:
            segment = features[:, start:start + frames_per_window]
            segments.append(segment)
            start += step_frames

        # Handle last segment if signal is too short
        if len(segments) == 0:
            if n_frames < frames_per_window:
                # Pad if too short
                padded = np.pad(features, ((0, 0), (0, frames_per_window - n_frames)),
                              mode='edge')
                segments.append(padded)
            else:
                segments.append(features[:, :frames_per_window])

        return np.array(segments)

    def process(self, audio_path, augment=False):
        """
        Complete preprocessing pipeline for audio

        Args:
            audio_path: Path to audio file
            augment: Whether to apply data augmentation (for training)

        Returns:
            Preprocessed MFCC segments (num_segments x n_features x frames_per_window)
        """
        # Extract MFCC
        mfcc = self.extract_mfcc(audio_path, augment=augment)

        if mfcc is None:
            return None

        # Normalize
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        mfcc_normalized = (mfcc - mean) / (std + 1e-8)

        # Segment
        segmented = self.segment_features(mfcc_normalized)

        return segmented


class AudioAugmentation:
    """
    Audio augmentation techniques for training data
    """

    def __init__(self, sr=16000):
        self.sr = sr

    def time_stretch(self, y, rate=1.0):
        """Time stretch audio by a factor"""
        return librosa.effects.time_stretch(y, rate=rate)

    def pitch_shift(self, y, n_steps=0):
        """Pitch shift audio by n_steps semitones"""
        return librosa.effects.pitch_shift(y, sr=self.sr, n_steps=n_steps)

    def add_noise(self, y, noise_factor=0.005):
        """Add random noise to audio"""
        noise = np.random.normal(0, 1, len(y))
        return y + noise_factor * noise

    def volume_augment(self, y, factor=1.0):
        """Adjust volume by a factor"""
        return y * factor

    def augment_audio(self, y):
        """Apply random augmentation to audio"""
        # Random time stretch (0.8x to 1.2x)
        rate = np.random.uniform(0.8, 1.2)
        y = self.time_stretch(y, rate)

        # Random pitch shift (-2 to +2 semitones)
        pitch_steps = np.random.uniform(-2, 2)
        y = self.pitch_shift(y, pitch_steps)

        # Random volume adjustment (0.7x to 1.3x)
        volume_factor = np.random.uniform(0.7, 1.3)
        y = self.volume_augment(y, volume_factor)

        # Random noise injection (5% of max amplitude)
        if np.random.random() < 0.5:  # 50% chance
            noise_factor = np.random.uniform(0, 0.05)
            y = self.add_noise(y, noise_factor)

        return y
