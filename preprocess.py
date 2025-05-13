
"""Preprocessing utilities for cavitation detection.
Implements band-pass filtering, signal normalization and segmentation.

All methods are encapsulated in the Preprocessor class to comply with the
project's class‑only design guideline.
"""
import numpy as np
from scipy.signal import butter, sosfilt

class Preprocessor:
    """Signal preprocessing for cavitation analysis."""

    def __init__(self, fs: float, band: tuple = (1e5, 4.5e5), seg_length: float = 0.1):
        """
        Parameters
        ----------
        fs : float
            Sampling frequency in Hz.
        band : tuple
            (low, high) cut‑off frequencies for the band‑pass filter.
        seg_length : float
            Segment length in seconds.
        """
        self.fs = fs
        self.band = band
        self.seg_length = seg_length
        self.sos = butter(8, [band[0] / (0.5 * fs), band[1] / (0.5 * fs)],
                          btype='bandpass', output='sos')

    def bandpass(self, signal: np.ndarray) -> np.ndarray:
        """Apply zero‑phase band‑pass filter."""
        return sosfilt(self.sos, signal)

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """Zero‑mean, unit‑variance normalization."""
        eps = np.finfo(float).eps
        return (signal - np.mean(signal)) / (np.std(signal) + eps)

    def segment(self, signal: np.ndarray) -> np.ndarray:
        """Split signal in overlapping segments.

        Returns
        -------
        np.ndarray
            Shape (n_segments, segment_samples)
        """
        seg_samples = int(self.seg_length * self.fs)
        step = seg_samples  # non‑overlapping by default
        n_segments = max((len(signal) - seg_samples) // step + 1, 1)
        return np.stack([
            signal[i * step: i * step + seg_samples]
            for i in range(n_segments)
        ])
